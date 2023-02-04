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
//采样hizmap使用自己的mipmap
const float4 Uv_Bias_Lut[9]={float4(0,0.333333334,1.0,1.0),
                            float4(0.0,0.0,0.5,0.333333333),
                            float4(0.5,0.16666667,0.75,0.333333333),
                            float4(0.75,0.25,0.875,333333333),
                            float4(0.875,0.2916666667,0.9375,0.333333333),
                            float4(0.9375,0.3125,0.96875,0.333333333),
                            float4(0.96875,0.3229166666667,0.984375,0.333333333),
                            float4(0.984375,0.328125,0.9921875,0.333333333),
                            float4(0.99609375,0.33072916666667,0.998046875,0.333333333)};
float2 MipmapUv_Bias(float2 uv,int mipmaplevel){
    float4 One=Uv_Bias_Lut[mipmaplevel];
    return float2(lerp(One.x,One.z,uv.x),lerp(One.y,One.w,uv.y));
}
//declaredepth这个库，指定lod等级
TEXTURE2D_X_FLOAT(_CameraDepthTexture);
SamplerState Point_Clamp;
SamplerState Linear_Clamp;
half SampleSceneDepth(half2 uv)
{
    return SAMPLE_TEXTURE2D_X_LOD(_CameraDepthTexture, Point_Clamp, uv,0).r;
}
//*******************************
//declarenormal这个库，指定lod等级
TEXTURE2D_X_FLOAT(_CameraNormalsTexture);
SAMPLER(sampler_CameraNormalsTexture);

half3 SampleSceneNormals(half2 uv)
{
    half3 normal = SAMPLE_TEXTURE2D_X_LOD(_CameraNormalsTexture, sampler_CameraNormalsTexture, UnityStereoTransformScreenSpaceTex(uv),0).xyz;

    #if defined(_GBUFFER_NORMALS_OCT)
    half2 remappedOctNormalWS = Unpack888ToFloat2(normal); // values between [ 0,  1]
    half2 octNormalWS = remappedOctNormalWS.xy * 2.0 - 1.0;    // values between [-1, +1]
    normal = UnpackNormalOctQuadEncode(octNormalWS);
    #endif

    return normal;
}
//*******************************

//低差异序列
float RadicalInverse_Vdc(uint bits)
{
    bits = (bits << 16u) | (bits >> 16u);
    bits = ((bits & 0x55555555u) << 1u) | ((bits & 0xAAAAAAAAu) >> 1u);
    bits = ((bits & 0x33333333u) << 2u) | ((bits & 0xCCCCCCCCu) >> 2u);
    bits = ((bits & 0x0F0F0F0Fu) << 4u) | ((bits & 0xF0F0F0F0u) >> 4u);
    bits = ((bits & 0x00FF00FFu) << 8u) | ((bits & 0xFF00FF00u) >> 8u);
    return float(bits) * 2.3283064365386963e-10; // / 0x100000000 
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

half TemporalFilterIntensity;

int MaxMipLevel;
TEXTURE2D_X_FLOAT(HiZMipmap_Level_0);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_1);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_2);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_3);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_4);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_5);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_6);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_7);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_8);

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
half GetEyeDepth_HiZBuffer(Texture2D HizBuffer[9],half2 uv,int MipLevel){
    half ZbufferDepth=SAMPLE_TEXTURE2D_X_LOD(HizBuffer[MipLevel],Point_Clamp,uv,0).r;
    return LinearEyeDepth(ZbufferDepth,_ZBufferParams);
}
half GetEyeDepth_NoFilter(half2 uv){
    return LinearEyeDepth(SampleSceneDepth(uv),_ZBufferParams);
}
half Pow2(half x){
    return x*x;
}
half pow4(half x){
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
half3 GetPositionVs_HiZBuffer(Texture2D HizBuffer[9],half2 uv,int a){
    half Zbuffer=SAMPLE_TEXTURE2D_X_LOD(HizBuffer[a],Point_Clamp,uv,0).r;
    return GetPositionVs(Zbuffer,uv);
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
half GGX_D(half NoH,half Roughness){
    half m = Roughness * Roughness;
	half m2 = m * m;
    half d = (m2*NoH-NoH)*NoH+1;
	return m2 / (3.1415926 * d * d);
}
half GGX_G(half NoL, half NoV, half Roughness){
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

	half D = GGX_D(NoH, Roughness);
	half G = GGX_G(NoL, NoV, Roughness);
    half F = pow4(1.0-NoV);//unity抄的
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
			
	half D = GGX_D(CosTheta,Roughness);
			
	Pdf_D = D * CosTheta;

	return H;
}
struct RayTraceResult{
    half3 SceneColor;
    half HitZBufferdepth;
    half2 Hituv;
    bool HitMask;
};
RayTraceResult RayTrace_HierarchyZ(Texture2D HiZBuffer[9],Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,
                            half2 uv,half3 RayOriginVs,half2 TexelSize,half StepLength,half Thickness,uint RandomSeed){//TexelSize横纵像素数
    RayTraceResult Result_None;
    Result_None.SceneColor=0;
    Result_None.HitZBufferdepth=0;
    Result_None.Hituv=0;
    Result_None.HitMask=false;
    RayTraceResult Result;
    Result=Result_None;

    int Miplevel_Cur=0;
    
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
    half MinLengthSS=length(rcp(TexelSize.xy)*length(RayDirSS.xy)/abs(RayDirSS.x));
    half Random_Step=lerp(0.6,1.4,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed*2))));
    half s=max(MinLengthSS,StepLength*Random_Step)/L;
    int Max_StepCount=1+(int)rcp(s*L);
    half2 RayDirSS_PerStep=RayDirSS*s;
    half2 Ray;
    half z0=RayOriginVs.z;
    half z1=RayEndVs_Max.z;
    half T=PerspectiveCorrectInterpolateCoefficient_VS(s,z0,z1);
    z1=z0+T*(z1-z0);
    half DeltaZ=z0;
    half Z_Pre=z0;
    int i=0;

    while(i<=Max_StepCount){
        i++;
        Ray=RayOriginSS+RayDirSS_PerStep*i;
        // UNITY_BRANCH
        // if(Ray.x>1.0 || Ray.x<0.0 || Ray.y>1.0 || Ray.y<0.0){return Result_None;}
        half t=PerspectiveCorrectInterpolateCoefficient_VS(i,z0,z1);
        half z=z0*(1-t)+z1*t;
        // UNITY_BRANCH
        // if(-z>F || -z<N){return Result_None;}
        DeltaZ=z-Z_Pre;
        Z_Pre=z;
        half SampleDepth=GetEyeDepth_HiZBuffer(HiZBuffer,Ray.xy,Miplevel_Cur);
        half Delta=-z-SampleDepth;
        bool IsHit=Delta>0 && Delta<Thickness;
        bool IsLevel0=Miplevel_Cur==0;

        UNITY_BRANCH
        if(IsHit){
            if(IsLevel0){
                i--;
                Miplevel_Cur--;
                RayDirSS_PerStep/=2.0;
                break;
            }
            else{
                if(dot(normalize(RayDirVs),GetNormalVs(Ray.xy))>0.0){return Result_None;}
                Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,Ray.xy,0).xyz;
                Result.Hituv=Ray.xy;
                Result.HitZBufferdepth=EyeDepthToZbuffer(-z);
                Result.HitMask=true;
                return Result;
            }
        }
        else{
            Miplevel_Cur++;
            RayDirSS_PerStep*=2.0;
            break;
        }
    }
    return Result_None;
}
RayTraceResult RayTrace_Linear(Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,
                            half2 uv,half3 RayOriginVs,half2 TexelSize,half StepLength,half Thickness,uint RandomSeed){//TexelSize横纵像素数
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
    half s=max(MinLengthSS,StepLength*Random_Step)/L;
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
        UNITY_BRANCH
        if(-z>F || -z<N){return Result_None;}
        DeltaZ=z-Z_Pre;
        Z_Pre=z;
        half SampleDepth=GetEyeDepth(Ray.xy);
        half Delta=-z-SampleDepth;
        #if defined BINARY_SEARCH
        UNITY_BRANCH
        if(Delta>0&& Delta<Thickness*BinarySearch_Tolerance){
            half BinarySearchLength=0.5f;
            half2 Ray_Bin;
            half z_Bin;
            half IterationsScale=0.25f;
            half Delta_Bin;
            UNITY_LOOP
            for(int j=0;j<1+BINARYSEARCHITERATIONS;j++){
                half S_Bin=i-1.0f+BinarySearchLength;
                Ray_Bin=RayOriginSS+RayDirSS_PerStep*S_Bin;
                half T_Bin=PerspectiveCorrectInterpolateCoefficient_VS(S_Bin,z0,z1);
                z_Bin=z0+T_Bin*(z1-z0);
                half SampleDepth_Bin=GetEyeDepth(Ray_Bin.xy);
                Delta_Bin=-z_Bin-SampleDepth_Bin;
                UNITY_BRANCH
                if(Delta_Bin>0.0f){//超过
                    BinarySearchLength-=IterationsScale;
                }
                else{//没超过
                    BinarySearchLength+=IterationsScale;
                }
                IterationsScale*=0.5f;
            }
            UNITY_BRANCH
            if(dot(normalize(RayDirVs),GetNormalVs(Ray_Bin.xy))>0.0){return Result_None;}
            Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,Ray_Bin.xy,0).xyz;
            Result.Hituv=Ray_Bin.xy;
            Result.HitZBufferdepth=EyeDepthToZbuffer(-z_Bin);
            Result.HitMask=true;
            return Result;
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

    Texture2D HiZBuffer[9];
    HiZBuffer[0]=HiZMipmap_Level_0;
    HiZBuffer[1]=HiZMipmap_Level_1;
    HiZBuffer[2]=HiZMipmap_Level_2;
    HiZBuffer[3]=HiZMipmap_Level_3;
    HiZBuffer[4]=HiZMipmap_Level_4;
    HiZBuffer[5]=HiZMipmap_Level_5;
    HiZBuffer[6]=HiZMipmap_Level_6;
    HiZBuffer[7]=HiZMipmap_Level_7;
    HiZBuffer[8]=HiZMipmap_Level_8;

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
        half3 RayDir=GetReflectDir(ViewDir,NormalVs);
        RayTraceResult RayTraceResult_one=RayTrace_HierarchyZ(HiZBuffer,SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,STEPLENGTH,THICKNESS,1);
        Result.SceneColor_Pdf.xyz=RayTraceResult_one.SceneColor;
        Result.HitPoint_Mask.xyz=half3(RayTraceResult_one.Hituv,RayTraceResult_one.HitZBufferdepth);
        Result.SceneColor_Pdf.w=1;
        Result.HitPoint_Mask.w=RayTraceResult_one.HitMask;
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

TEXTURE2D_X_FLOAT(_MotionVectorTexture);
SAMPLER(sampler_MotionVectorTexture);

TEXTURE2D(RT_SSR_Denoised_Pre);
SAMPLER(sampler_RT_SSR_Denoised_Pre);
TEXTURE2D(RT_SSR_TemporalFilter_In);
SAMPLER(sampler_RT_SSR_TemporalFilter_In);

half2 GetClosestUv(half2 uv){//要使用去除抖动的uv
	half2 Closest_Offset=half2(0,0);
	UNITY_UNROLL
	for(int i=-1;i<=1;i++){
		UNITY_UNROLL
		for(int j=-1;j<=1;j++){
			int flag=step(GetEyeDepth_NoFilter(uv),GetEyeDepth_NoFilter(uv+RT_TexelSize.xy*half2(i,j)));
			Closest_Offset=lerp(Closest_Offset,half2(i,j),flag);
		}
	}
	return RT_TexelSize.xy*Closest_Offset+uv;
}
half3 RGB_YCoCg(half3 c){
    return half3(
            c.x/4.0 + c.y/2.0 + c.z/4.0,
            c.x/2.0 - c.z/2.0,
        -c.x/4.0 + c.y/2.0 - c.z/4.0
    );
}
half3 YCoCg_RGB(half3 c){
    return saturate(half3(
        c.x + c.y - c.z,
        c.x + c.z,
        c.x - c.y - c.z
    ));
}
half4 SampleSSRColor(Texture2D SSR_Color,half2 uv){//采样并转换为ycocg
    half4 rgba=SAMPLE_TEXTURE2D(SSR_Color,Point_Clamp,uv).xyzw;
    return half4(RGB_YCoCg(rgba.xyz),rgba.w);
}
void GetBoundingBox(out half4 cmin,out half4 cmax,out half4 cavg,half2 uv){
    half2 du=half2(RT_TexelSize.x,0.0);
    half2 dv=half2(0.0,RT_TexelSize.y);

    half4 ctl = SampleSSRColor(RT_SSR_Denoised_Pre, uv - dv - du);
    half4 ctc = SampleSSRColor(RT_SSR_Denoised_Pre, uv - dv);
    half4 ctr = SampleSSRColor(RT_SSR_Denoised_Pre, uv - dv + du);
    half4 cml = SampleSSRColor(RT_SSR_Denoised_Pre, uv - du);
    half4 cmc = SampleSSRColor(RT_SSR_Denoised_Pre, uv);
    half4 cmr = SampleSSRColor(RT_SSR_Denoised_Pre, uv + du);
    half4 cbl = SampleSSRColor(RT_SSR_Denoised_Pre, uv + dv - du);
    half4 cbc = SampleSSRColor(RT_SSR_Denoised_Pre, uv + dv);
    half4 cbr = SampleSSRColor(RT_SSR_Denoised_Pre, uv + dv + du);

    cmin = min(ctl, min(ctc, min(ctr, min(cml, min(cmc, min(cmr, min(cbl, min(cbc, cbr))))))));
	cmax = max(ctl, max(ctc, max(ctr, max(cml, max(cmc, max(cmr, max(cbl, max(cbc, cbr))))))));
    cavg = (ctl + ctc + ctr + cml + cmc + cmr + cbl + cbc + cbr) / 9.0;

    half4 cmin5 = min(ctc, min(cml, min(cmc, min(cmr, cbc))));
    half4 cmax5 = max(ctc, max(cml, max(cmc, max(cmr, cbc))));
    half4 cavg5 = (ctc + cml + cmc + cmr + cbc) / 5.0;
    cmin = 0.5 * (cmin + cmin5);
    cmax = 0.5 * (cmax + cmax5);
    cavg = 0.5 * (cavg + cavg5);

    half2 chroma_extent = 0.25 * 0.5 * (cmax.r - cmin.r);
    half2 chroma_center = cmc.gb;
    cmin.yz = chroma_center - chroma_extent;
    cmax.yz = chroma_center + chroma_extent;
    cavg.yz = chroma_center;
}

half4 ClipToBoundingBox(half4 AABB_Min,half4 AABB_Max,half4 AABB_Avg,half4 SSR_Color_Pre){
    AABB_Avg=clamp(AABB_Avg,AABB_Min,AABB_Max);
    float4 r = SSR_Color_Pre - AABB_Avg;
    float3 rmax = AABB_Max.xyz - AABB_Avg.xyz;
    float3 rmin = AABB_Min.xyz - AABB_Avg.xyz;

    const float eps = FLT_EPS;
    UNITY_BRANCH
    if (r.x > rmax.x + eps)
        r *= (rmax.x / r.x);
    UNITY_BRANCH
    if (r.y > rmax.y + eps)
        r *= (rmax.y / r.y);
    UNITY_BRANCH
    if (r.z > rmax.z + eps)
        r *= (rmax.z / r.z);
    UNITY_BRANCH
    if (r.x < rmin.x - eps)
        r *= (rmin.x / r.x);
    UNITY_BRANCH
    if (r.y < rmin.y - eps)
        r *= (rmin.y / r.y);
    UNITY_BRANCH
    if (r.z < rmin.z - eps)
        r *= (rmin.z / r.z);

    return AABB_Avg + r;
}

half4 Frag_StochasticSSR_TemporalFilter(VertexOutput i):SV_Target{
    half2 uv=i.uv;
    UNITY_BRANCH
    if(GetSkyBoxMask(uv)){return 0.0;}
    half2 Closest_uv=GetClosestUv(i.uv);
    half2 Velocity=SAMPLE_TEXTURE2D_X(_MotionVectorTexture,sampler_MotionVectorTexture,Closest_uv).rg;

    half4 SSR_Color_Pre=SampleSSRColor(RT_SSR_Denoised_Pre,uv-Velocity);
	half4 SSR_Color=SampleSSRColor(RT_SSR_TemporalFilter_In,uv);

    half4 AABB_Min,AABB_Max,AABB_Avg;
    GetBoundingBox(AABB_Min,AABB_Max,AABB_Avg,uv);
    half4 SSR_Color_Pre_Clipped=ClipToBoundingBox(AABB_Min,AABB_Max,AABB_Avg,SSR_Color_Pre);
    
    half Temporal_BlendWeight=saturate(TemporalFilterIntensity*(1-length(Velocity)*8));
	half4 SSR_Denoised=lerp(SSR_Color,SSR_Color_Pre,Temporal_BlendWeight);
    SSR_Denoised=half4(YCoCg_RGB(SSR_Denoised.xyz),SSR_Denoised.w);

    return SSR_Denoised;
}