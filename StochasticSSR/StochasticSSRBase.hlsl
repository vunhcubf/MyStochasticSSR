#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityGBuffer.hlsl"
#include "Packages/com.unity.render-pipelines.core/ShaderLibrary/Random.hlsl"

#if defined FULL_PRECISION_SSR
#define half float
#define half2 float2
#define half3 float3
#define half4 float4
#define half4x4 float4x4
#define half3x3 float3x3
#endif

//TEXTURE2D(unity_SpecCube0);
//declaredepth这个库，指定lod等级
TEXTURE2D_X_FLOAT(_CameraDepthTexture);
SamplerState Point_Clamp;
SamplerState Linear_Clamp;
half SampleSceneDepth(half2 uv)
{
    return SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, Point_Clamp, UnityStereoTransformScreenSpaceTex(uv),0).r;
}
//*******************************
//declarenormal这个库，指定lod等级
TEXTURE2D_X_FLOAT(_CameraNormalsTexture);
SAMPLER(sampler_CameraNormalsTexture);

float3 SampleSceneNormals(float2 uv)
{
    float3 normal = SAMPLE_TEXTURE2D_X_LOD(_CameraNormalsTexture, sampler_CameraNormalsTexture, UnityStereoTransformScreenSpaceTex(uv),0).xyz;

    #if defined(_GBUFFER_NORMALS_OCT)
    float2 remappedOctNormalWS = Unpack888ToFloat2(normal); // values between [ 0,  1]
    float2 octNormalWS = remappedOctNormalWS.xy * 2.0 - 1.0;    // values between [-1, +1]
    normal = UnpackNormalOctQuadEncode(octNormalWS);
    #endif

    return normal;
}
//*******************************

half4x4 World2View_Matrix;
half4x4 View2World_Matrix;
half4x4 InvProjection_Matrix;
half4x4 Projection_Matrix;
half fov;
half F,N;

half MirrorReflectionThreshold;
half ANGLE_BIAS;
half MAXROUGHNESS;
int BINARYSEARCHITERATIONS;

static const int2 Pixel_Bias[9]={int2(0,-1),int2(0,0),int2(0,1),int2(1,-1),int2(1,0),int2(1,1),int2(-1,-1),int2(-1,0),int2(-1,1)};

half Bilinear_Tolerance;
half BinarySearch_Tolerance;
half THICKNESS;
half STEPLENGTH;
int SAMPLESPERPIXEL;

TEXTURE2D(SSR_CameraTexture);
TEXTURE2D_X_FLOAT(RT_SSR_Result1);
TEXTURE2D_X_FLOAT(RT_SSR_Result2);

TEXTURE2D(_GBuffer0);
SAMPLER(sampler_GBuffer0);
TEXTURE2D(_GBuffer1);
SAMPLER(sampler_GBuffer1);
TEXTURE2D(_GBuffer2);
SAMPLER(sampler_GBuffer2);

half4 RT_TexelSize;

half EyeDepthToZbuffer(half eyedepth){
    return ((1/eyedepth)-_ZBufferParams.w)/_ZBufferParams.z;
}
half SampleEyeDepthBilinear(half2 uv){
    half2 uv_raw=uv;

    uv+=half2(-0.5,-0.5)/RT_TexelSize.zw;
    half2 uv_frac=frac(uv*RT_TexelSize.zw);
    half2 uv_00,uv_01,uv_10,uv_11;
    uv_00=uv;
    uv_01=uv_00+half2(0,1)/RT_TexelSize.zw;
    uv_10=uv_00+half2(1,0)/RT_TexelSize.zw;
    uv_11=uv_00+half2(1,1)/RT_TexelSize.zw;

    half EyeDepth_00,EyeDepth_01,EyeDepth_10,EyeDepth_11;
    EyeDepth_00=LinearEyeDepth(SampleSceneDepth(uv_00),_ZBufferParams);
    EyeDepth_01=LinearEyeDepth(SampleSceneDepth(uv_01),_ZBufferParams);
    EyeDepth_10=LinearEyeDepth(SampleSceneDepth(uv_10),_ZBufferParams);
    EyeDepth_11=LinearEyeDepth(SampleSceneDepth(uv_11),_ZBufferParams);
    
    half EyeDepth_Min;
    EyeDepth_Min=min(EyeDepth_00,min(EyeDepth_01,min(EyeDepth_10,EyeDepth_11)));

    int Out_Count=0;
    Out_Count=(EyeDepth_00-EyeDepth_Min>=Bilinear_Tolerance)?Out_Count+1:Out_Count;
    Out_Count=(EyeDepth_01-EyeDepth_Min>=Bilinear_Tolerance)?Out_Count+1:Out_Count;
    Out_Count=(EyeDepth_10-EyeDepth_Min>=Bilinear_Tolerance)?Out_Count+1:Out_Count;
    Out_Count=(EyeDepth_11-EyeDepth_Min>=Bilinear_Tolerance)?Out_Count+1:Out_Count;
    UNITY_BRANCH
    if(Out_Count==0){
        half linear_sample=EyeDepth_00*(1-uv_frac.x)*(1-uv_frac.y)+EyeDepth_10*uv_frac.x*(1-uv_frac.y)+EyeDepth_01*uv_frac.y*(1-uv_frac.x)+EyeDepth_11*uv_frac.y*uv_frac.x;
        return linear_sample;
    }
    else{
        return LinearEyeDepth(SampleSceneDepth(uv_raw),_ZBufferParams);
    }
}
half GetEyeDepth(half2 uv){
    #if defined BILINEAR_DEPTHBUFFER
    return SampleEyeDepthBilinear(uv);
    #else
    return LinearEyeDepth(SampleSceneDepth(uv),_ZBufferParams);
    #endif
}
half GetEyeDepth_NoFilter(half2 uv){
    return LinearEyeDepth(SampleSceneDepth(uv),_ZBufferParams);
}
half Pow2(half x){
    return x*x;
}
half Pow4(half x){
    half Out=Pow2(x);
    Out=Pow2(Out);
    return Out;
}
half3 GetReflectDir(half3 ViewDir,half3 normalVS){
    half cos=dot(ViewDir,normalVS);
    return normalize(2*normalVS*cos-ViewDir);
}
bool GetSkyBoxMask(half2 uv){
    return step(SampleSceneDepth(uv),0);
}
half3 GetPositionVs(half NDCDepth,half2 uv){//不支持reversed-z
    half3 Vec;
    half tangent=tan(fov*3.1415926/360.0);
    Vec.xy=uv*2-1;
    Vec.xy*=half2(_ScreenParams.x/_ScreenParams.y,1);
    Vec.z=-1/tangent;
    return Vec*LinearEyeDepth(NDCDepth,_ZBufferParams)*tangent;
}
half3 GetPositionVs(half2 uv,half EyeDepth){//不支持reversed-z
    half3 Vec;
    half tangent=tan(fov*3.1415926/360.0);
    Vec.xy=uv*2-1;
    Vec.xy*=half2(_ScreenParams.x/_ScreenParams.y,1);
    Vec.z=-1/tangent;
    return Vec*EyeDepth*tangent;
}
half3 GetPositionVs(half2 uv){
    #if defined BILINEAR_DEPTHBUFFER
    return GetPositionVs(uv,SampleEyeDepthBilinear(uv));
    #else
    return GetPositionVs(SampleSceneDepth(uv),uv);
    #endif
}
half3 GetPositionVs_NoFilter(half2 uv){
    return GetPositionVs(SampleSceneDepth(uv),uv);
}
half3 GetViewDir(half2 uv){
    half3 Vec;
    half tangent=tan(fov*3.1415926/360.0);
    Vec.xy=uv*2-1;
    Vec.xy*=half2(_ScreenParams.x/_ScreenParams.y,1);
    Vec.z=-1/tangent;
    return -normalize(Vec);
}
half3 GetPositionWs(half2 uv){
    half4 PositionVs=half4(GetPositionVs(uv),1);
    half4 PositionWs=mul(View2World_Matrix,PositionVs);
    return PositionWs.xyz/PositionWs.w;
}
half4 GetPositionCS(half3 In){
    return mul(Projection_Matrix,half4(In,1));
}
half3 GetPositionSS(half3 In){
    half4 Out=GetPositionCS(In);
    Out/=Out.w;
    Out.xy=Out.xy*0.5+0.5;
    return Out.xyz;
}
half3 GetPositionSS(half2 In){
    return GetPositionSS(GetPositionVs(In));
}
half3 GetNormalVs(half2 uv){
    half3 Norm=SampleSceneNormals(uv);
    Norm=mul((half3x3)World2View_Matrix,Norm);
    return normalize(Norm);
}
half PerspectiveCorrectInterpolateCoefficient_SS(half t,half z0,half z1){//z=z1*t+z0*(1-t)
    return -(t*z1)/(z1*t+(1-t)*z0);
}
half PerspectiveCorrectInterpolateCoefficient_VS(half s,half z0,half z1){//z=z1*t+z0*(1-t)
    return (z0*s)/(s*z0-z1*(1+s));
}
half D_GGX(half NoH,half Roughness){
    half m = Roughness * Roughness;
	half m2 = m * m;
    half d = (m2*NoH-NoH)*NoH+1;
	return m2 / (3.1415926 * d * d);
}
half G_GGX(half NoL, half NoV, half Roughness){
	half a = Pow2(Roughness);
	half LambdaL = NoV * (NoL * (1 - a) + a);
	half LambdaV = NoL * (NoV * (1 - a) + a);
	return (0.5 * rcp(LambdaV + LambdaL+1e-5));
}
half Brdf_GGX(half3 V, half3 L, half3 N, half Roughness){
	half3 H = normalize(L + V);

	half NoH = saturate(dot(N, H));
	half NoL = saturate(dot(N, L));
	half NoV = saturate(dot(N, V));

	half D = D_GGX(NoH, Roughness);
	half G = G_GGX(NoL, NoV, Roughness);
    half F = Pow4(1.0-NoV);//unity抄的
	return max(0,D*G);
}
half Brdf_Unity(half3 viewDirectionWS, half3 lightDirectionWS,half3 normalWS,half Roughness){//从unity抄过来的，更亮点
    half3 halfDir = normalize(lightDirectionWS + half3(viewDirectionWS));
    half NoH = saturate(dot(normalWS, halfDir));
    half LoH = saturate(dot(lightDirectionWS, halfDir));
    half Roughness2=Roughness*Roughness;
    half d = NoH * NoH * (Roughness2-1) + 1.00001f;
    half normalizationTerm=Roughness*4.0+2.0;
    half LoH2 = LoH * LoH;
    half specularTerm = Roughness2 / ((d * d) * max(0.1f, LoH2) * normalizationTerm);
    return specularTerm;
}
half3 ImportanceSampleGGX(half2 E, half Roughness,out half Pdf_D) {
    half m = Roughness * Roughness;
	half m2 = m * m;
	half Phi = 2 * 3.14 * E.x;
	half CosTheta = sqrt((1 - E.y) / ( 1 + (m2 - 1) * E.y));
	half SinTheta = sqrt(1 - CosTheta * CosTheta);

	half3 H;
	H.x = SinTheta * cos(Phi);
	H.y = SinTheta * sin(Phi);
	H.z = CosTheta;
			
	half D = D_GGX(CosTheta,Roughness);
			
	Pdf_D = D * CosTheta;

	return H;
}
struct RayTraceResult{
    half3 SceneColor;
    half HitZBufferdepth;
    half2 Hituv;
    bool HitMask;
};
RayTraceResult RayTrace_Linear(Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,half2 uv,half3 RayOriginVs,half2 TexelSize,half StepLength,half Thickness,uint RandomSeed){//TexelSize横纵像素数
    RayTraceResult Result_None;
    Result_None.SceneColor=0;
    Result_None.HitZBufferdepth=0;
    Result_None.Hituv=0;
    Result_None.HitMask=false;
    RayTraceResult Result;
    Result=Result_None;
    
    half Vertical_Extension;
    half Vertical_Extension_Plane;
    UNITY_BRANCH
    if(RayDirVs.z>0){
        Vertical_Extension=(-N-RayOriginVs.z)/RayDirVs.z;
        RayDirVs*=Vertical_Extension;
    }

    half Random_Thickness=lerp(0.6,1.4,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed))));
    Thickness*=Random_Thickness;

    half3 RayEndVs_Max=RayOriginVs+RayDirVs;
    half2 RayOriginSS=GetPositionSS(RayOriginVs).xy;
    half2 RayEndSS=GetPositionSS(RayEndVs_Max).xy;
    half2 RayDirSS=RayEndSS-RayOriginSS;

    half L=length(RayDirSS);
    half MinLengthSS=length(rcp(TexelSize.xy));
    half Random_Step=lerp(0.6,1.4,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed*2))));
    half s=max(0,StepLength*Random_Step)/L;
    int Max_StepCount=1+(int)rcp(s*L);
    half2 RayDirSS_PerStep=RayDirSS*s;
    
    half2 Ray;

    half z0=RayOriginVs.z;
    half z1=RayEndVs_Max.z;
    half T=PerspectiveCorrectInterpolateCoefficient_VS(s,z0,z1);
    z1=z0+T*(z1-z0);
    half DeltaZ=z0;
    half Z_Pre=z0;
    UNITY_LOOP
    for(int i=1;i<=Max_StepCount;i++){
        Ray=RayOriginSS+RayDirSS_PerStep*i;
        UNITY_BRANCH
        if(Ray.x>1.0 || Ray.x<0.0 || Ray.y>1.0 || Ray.y<0.0){return Result_None;}
        
        half t=PerspectiveCorrectInterpolateCoefficient_VS(i,z0,z1);
        half z=z0*(1-t)+z1*t;
        DeltaZ=z-Z_Pre;
        Z_Pre=z;
        half SampleDepth=GetEyeDepth(Ray.xy);
        half Delta=-z-SampleDepth;

        
        #if defined BINARY_SEARCH
        UNITY_BRANCH
        if(Delta>0&& Delta<Thickness*BinarySearch_Tolerance){
            half2 RayStartSS_Bin=Ray-RayDirSS_PerStep;
            half2 RayEndSS_Bin=Ray;
            half StartZ=z-DeltaZ;
            half EndZ=z;
            half MidZ;
            half Delta_Bin;
            half2 MidRay;
            UNITY_LOOP
            for(int j=0;j<1+BINARYSEARCHITERATIONS;j++){
                MidZ=lerp(StartZ,EndZ,PerspectiveCorrectInterpolateCoefficient_VS(0.5,StartZ,EndZ));
                MidRay=lerp(RayStartSS_Bin,RayEndSS_Bin,0.5);
                if(MidRay.x>1.0 || MidRay.x<0.0 || MidRay.y>1.0 || MidRay.y<0.0){return Result_None;}
                Delta_Bin=-MidZ-GetEyeDepth_NoFilter(MidRay);
                UNITY_BRANCH
                if(Delta_Bin>0){
                    RayEndSS_Bin=MidRay;
                    EndZ=MidZ;
                }
                else{
                    RayStartSS_Bin=MidRay;
                    StartZ=MidZ;
                }
            }
            UNITY_BRANCH
            if(dot(normalize(RayDirVs),GetNormalVs(MidRay.xy))>0.0){return Result_None;}
            UNITY_BRANCH
            if(Delta_Bin<Thickness){
                Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,MidRay.xy,0).xyz;
                Result.Hituv=MidRay.xy;
                Result.HitZBufferdepth=EyeDepthToZbuffer(-MidZ);
                Result.HitMask=true;
                return Result;
            }
            
        }


        #else
        UNITY_BRANCH
        if(Delta>0 && Delta<Thickness){
            UNITY_BRANCH
            if(dot(normalize(RayDirVs),GetNormalVs(Ray.xy))>0.0){return Result_None;}
            Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,Ray.xy,0).xyz;
            Result.Hituv=Ray.xy;
            Result.HitZBufferdepth=EyeDepthToZbuffer(-z);
            Result.HitMask=true;
            return Result;
        }
        #endif
    }
    return Result_None;
}
half3x3 GetTanToViewMatrix(half3 NormalVs){
    half3 Tangent=(abs(NormalVs.z)<0.01)?normalize(half3(0,-NormalVs.z/NormalVs.y,1)):normalize(half3(0,1,-NormalVs.y/NormalVs.z));
    half3 Bitangent=cross(Tangent,NormalVs);
    return half3x3(Tangent.x,Bitangent.x,NormalVs.x,
                    Tangent.y,Bitangent.y,NormalVs.y,
                    Tangent.z,Bitangent.z,NormalVs.z);
}
half3x3 GetViewToTanMatrix(half3 NormalVs){
    half3 Tangent=(abs(NormalVs.z)<0.01)?normalize(half3(0,-NormalVs.z/NormalVs.y,1)):normalize(half3(0,1,-NormalVs.y/NormalVs.z));
    half3 Bitangent=cross(Tangent,NormalVs);
    return half3x3(Tangent.x,Tangent.y,Tangent.z,
                    Bitangent.x,Bitangent.y,Bitangent.z,
                    NormalVs.x,NormalVs.y,NormalVs.z);
}

struct VertexInput{
    half4 positionOS:POSITION;
    half2 uv:TEXCOORD0;
};
struct VertexOutput{
    half4 position:SV_POSITION;
    half2 uv:TEXCOORD0;
};
VertexOutput Vert_PostProcessDefault(VertexInput v)
{
    VertexOutput o;
    VertexPositionInputs positionInputs = GetVertexPositionInputs(v.positionOS.xyz);
    o.position = positionInputs.positionCS;
    o.uv = v.uv;
    return o;
}
struct PixelOutput_SSRResult{
    half4 SceneColor_Pdf:SV_Target0; 
    half4 HitPoint_Mask:SV_Target1;
};
PixelOutput_SSRResult Frag_StochasticSSR_RayTrace(VertexOutput i){
    half2 uv=i.uv;
    PixelOutput_SSRResult Result;
    PixelOutput_SSRResult Result_None;
    
    Result_None.SceneColor_Pdf=half4(0.0f,0.0f,0.0f,0.0f);
    Result_None.HitPoint_Mask=half4(0.0f,0.0f,0.0f,0.0f);

    UNITY_BRANCH
    if(GetSkyBoxMask(uv)){return Result_None;}
    
    half Roughness=1-SAMPLE_TEXTURE2D(_GBuffer2,sampler_GBuffer2,uv).a;
    UNITY_BRANCH
    if(Roughness>MAXROUGHNESS){return Result_None;}
    half3 RayStart=GetPositionVs_NoFilter(uv);
    half3 ViewDir=GetViewDir(uv);
    half3 NormalVs=GetNormalVs(uv);

    half2 TexelSize=RT_TexelSize.zw;

    half3 result=0;
    UNITY_BRANCH
    if(Roughness>=MirrorReflectionThreshold){
        half3x3 Tan2View=GetTanToViewMatrix(NormalVs);
        half3x3 View2Tan=GetViewToTanMatrix(NormalVs);
        bool Any_Hit=false;
        int Count_Hit=0;

        UNITY_LOOP
        for(int i=1;i<=SAMPLESPERPIXEL;i++){
            half2 hash2;
            half Pdf_tmp;
            hash2.x=frac(GenerateHashedRandomFloat(uint4(uv*RT_TexelSize.zw,abs(_SinTime.x)*1000,1)));
            hash2.y=frac(GenerateHashedRandomFloat(uint4(uv*RT_TexelSize.zw,abs(_SinTime.x)*2000,3)));

            half3 ViewDir_Tan=mul(View2Tan,ViewDir);
            half3 MicroNormal=ImportanceSampleGGX(hash2,Roughness,Pdf_tmp).xyz;
            half3 RayDir_Tan=GetReflectDir(ViewDir_Tan,MicroNormal);
            RayDir_Tan.z=max(RayDir_Tan.z,ANGLE_BIAS);
            RayDir_Tan=normalize(RayDir_Tan);
            half3 RayDir=mul(Tan2View,RayDir_Tan);

            RayTraceResult RayTraceResult_one=RayTrace_Linear(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,STEPLENGTH,THICKNESS,i);
            UNITY_BRANCH
            if(RayTraceResult_one.HitMask){
                Result.SceneColor_Pdf.xyz+=RayTraceResult_one.SceneColor;
                Result.SceneColor_Pdf.w+=Pdf_tmp;
                Result.HitPoint_Mask.xyz+=half3(RayTraceResult_one.Hituv,RayTraceResult_one.HitZBufferdepth);
                Count_Hit++;
                Any_Hit=true;
            }
        }
        UNITY_BRANCH
        if(Any_Hit){
            Result.SceneColor_Pdf/=Count_Hit;
            Result.HitPoint_Mask.xyz/=Count_Hit;
        }
        Result.HitPoint_Mask.w=Any_Hit;
    }
    else{
        int Count_Hit=0;
        bool Any_Hit=false;
        half3 RayDir=GetReflectDir(ViewDir,NormalVs);
        UNITY_LOOP
        for(int i=1;i<=SAMPLESPERPIXEL;i++){
            RayTraceResult RayTraceResult_one=RayTrace_Linear(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,STEPLENGTH,THICKNESS,i);
            UNITY_BRANCH
            if(RayTraceResult_one.HitMask){
                Result.SceneColor_Pdf.xyz+=RayTraceResult_one.SceneColor;
                Result.HitPoint_Mask.xyz+=half3(RayTraceResult_one.Hituv,RayTraceResult_one.HitZBufferdepth);
                Count_Hit++;
                Any_Hit=true;
            }
        }
        UNITY_BRANCH
        if(Any_Hit){
            Result.SceneColor_Pdf.xyz/=Count_Hit;
            Result.HitPoint_Mask.xyz/=Count_Hit;
        }
        Result.SceneColor_Pdf.w=1;
        Result.HitPoint_Mask.w=Any_Hit;
    }
    return Result;
}
void GetRayTraceResult(half2 uv,out RayTraceResult RayInfo,out half Pdf){
    half4 RT_1=SAMPLE_TEXTURE2D_X_LOD(RT_SSR_Result1,Point_Clamp,uv,0).xyzw;
    half4 RT_2=SAMPLE_TEXTURE2D_X_LOD(RT_SSR_Result2,Point_Clamp,uv,0).xyzw;
    RayInfo.SceneColor=RT_1.xyz;
    Pdf=RT_1.w;
    RayInfo.HitZBufferdepth=RT_2.z;
    RayInfo.Hituv=RT_2.xy;
    RayInfo.HitMask=RT_2.w;
}
half4 Frag_StochasticSSR_Resolve(VertexOutput i):SV_Target{
    half2 uv=i.uv;

    UNITY_BRANCH
    if(GetSkyBoxMask(uv)){return 0.0;}
    half3 positionVs=GetPositionVs_NoFilter(uv);
    half3 ViewDir=GetViewDir(uv);
    half3 NormalVs=GetNormalVs(uv);
    half Roughness=1-SAMPLE_TEXTURE2D(_GBuffer2,sampler_GBuffer2,uv).a;
    UNITY_BRANCH
    if(Roughness<MirrorReflectionThreshold){return half4(SAMPLE_TEXTURE2D_LOD(RT_SSR_Result1,Point_Clamp,uv,0).xyz,1);}

    half3 MC_Iteration=0;
    half Count_Hit=0;
    bool Any_Hit=false;
    UNITY_UNROLL
    for(int i=0;i<9;i++){
        half2 Offsetuv=RT_TexelSize.xy*Pixel_Bias[i];
        half2 NeighbourUv=uv+Offsetuv;

        RayTraceResult RayTrace_Info_One;
        half Pdf_One;
        GetRayTraceResult(NeighbourUv,RayTrace_Info_One,Pdf_One);
        UNITY_BRANCH
        if(RayTrace_Info_One.HitMask){
            half3 HitPosVs=GetPositionVs(RayTrace_Info_One.HitZBufferdepth,RayTrace_Info_One.Hituv);
            half3 RayDir=normalize(HitPosVs-positionVs);
            half Brdf=Brdf_GGX(ViewDir,RayDir,NormalVs,Roughness);
            half3 RenderEquation_Glossy=RayTrace_Info_One.SceneColor*Brdf*saturate(dot(NormalVs,RayDir));//不要忘记最后一项
            MC_Iteration+=RenderEquation_Glossy/max(Pdf_One,1e-3);
            Count_Hit=Count_Hit+1.0;
            Any_Hit=true;
        }
    }
    half3 SSR_Color=0;
    UNITY_BRANCH
    if(Any_Hit){
        SSR_Color=MC_Iteration/Count_Hit;
    }
    return half4(SSR_Color,1);
}