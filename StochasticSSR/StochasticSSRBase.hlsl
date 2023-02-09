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
    half3 normal = SAMPLE_TEXTURE2D_X_LOD(_CameraNormalsTexture, Point_Clamp, uv,0).xyz;

    #if defined(_GBUFFER_NORMALS_OCT)
    half2 remappedOctNormalWS = Unpack888ToFloat2(normal); // values between [ 0,  1]
    half2 octNormalWS = remappedOctNormalWS.xy * 2.0 - 1.0;    // values between [-1, +1]
    normal = UnpackNormalOctQuadEncode(octNormalWS);
    #endif

    return normalize(normal);
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
half Aspect;

half MirrorReflectionThreshold;
half ANGLE_BIAS;
half MAXROUGHNESS;
int BINARYSEARCHITERATIONS;

half TemporalFilterIntensity;

int MaxMipLevel;
TEXTURE2D_X_FLOAT(HiZMipmap_WithMipmap);

static const int2 Pixel_Bias[9]={int2(0,-1),int2(0,0),int2(0,1),int2(1,-1),int2(1,0),int2(1,1),int2(-1,-1),int2(-1,0),int2(-1,1)};

half Bilinear_Tolerance;
half BinarySearch_Tolerance;
half THICKNESS;
int MaxIterations;

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
half2 RT_TexelSize_HiZBuffer;
half RayBias;

static float2 MipmapUv_Bias(float2 uv,int Miplevel){
    float4 Uv_Bias_Lut0[9]={float4(0.0f,0.333333334f,1.0f,1.0f),
                            float4(0.0f,0.0f,0.5f,0.333333333f),
                            float4(0.5f,0.16666667f,0.75f,0.333333333f),
                            float4(0.75f,0.25f,0.875f,0.333333333f),
                            float4(0.875f,0.2916666667f,0.9375f,0.333333333f),
                            float4(0.9375f,0.3125f,0.96875f,0.333333333f),
                            float4(0.96875f,0.3229166666667f,0.984375f,0.333333333f),
                            float4(0.984375f,0.328125f,0.9921875f,0.333333333f),
                            float4(0.9921875f,0.33072916666667f,0.99609375f,0.333333333f)};
    float4 Uv_Bias_Lut=Uv_Bias_Lut0[Miplevel];
    uv=saturate(uv);
    return float2(lerp(Uv_Bias_Lut.x,Uv_Bias_Lut.z,uv.x),lerp(Uv_Bias_Lut.y,Uv_Bias_Lut.w,uv.y));
}
static float2 MipmapUv_Bias(float2 uv,float4 Uv_Bias_Lut){
    uv=saturate(uv);
    return float2(lerp(Uv_Bias_Lut.x,Uv_Bias_Lut.z,uv.x),lerp(Uv_Bias_Lut.y,Uv_Bias_Lut.w,uv.y));
}
static half EyeDepthToZbuffer(half eyedepth){
    return ((1/eyedepth)-_ZBufferParams.w)/_ZBufferParams.z;
}
static half SampleEyeDepthBilinear(half2 uv){
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
    [branch]
    if(Out_Count==0){
        half linear_sample=EyeDepth_00*(1-uv_frac.x)*(1-uv_frac.y)+EyeDepth_10*uv_frac.x*(1-uv_frac.y)+EyeDepth_01*uv_frac.y*(1-uv_frac.x)+EyeDepth_11*uv_frac.y*uv_frac.x;
        return linear_sample;
    }
    else{
        return LinearEyeDepth(SampleSceneDepth(uv_raw),_ZBufferParams);
    }
}
half SampleZBufferDepthBilinear(half2 uv){
    return EyeDepthToZbuffer(SampleEyeDepthBilinear(uv));
}
half GetEyeDepth(half2 uv){
    #if defined BILINEAR_DEPTHBUFFER
    return SampleEyeDepthBilinear(uv);
    #else
    return LinearEyeDepth(SampleSceneDepth(uv),_ZBufferParams);
    #endif
}
half GetEyeDepth_HiZBuffer(half2 uv,half4 HiZBuffer_Bias_Lut){
    half2 Uv_Biased=MipmapUv_Bias(uv,HiZBuffer_Bias_Lut);
    half ZbufferDepth=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_WithMipmap,Point_Clamp,Uv_Biased,0).r;
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
half pow8(half x){
    half Out=Pow2(x);
    Out=Pow2(Out);
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
half3 GetPositionCS(half3 In){ 
    half cot=rcp(tan(fov*3.1415926/360.0));
    half A_rcp=rcp(Aspect);
    return half3(-cot*A_rcp*In.x/In.z,-cot*In.y/In.z,EyeDepthToZbuffer(-In.z));
}
half3 GetPositionSS(half3 In){
    half3 Out=GetPositionCS(In);
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
    return (z0*s)/(z1-s*(z1-z0));
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
int2 RayToPixelPosition(half2 uv,half2 texelsize){
    return (int2)round(floor(uv.xy*texelsize));
}
struct RayTraceResult{
    half3 SceneColor;
    half HitZBufferdepth;
    half2 Hituv;
    bool HitMask;
};
half2 GetCurCell(half3 Ray,half2 CellSize){//获取的是格子左下角的坐标
    half2 Result=floor(Ray.xy*CellSize+(1e-3f).xx*rcp(CellSize.xy))/CellSize;
    return Result;
}
half AlignedCellXMarching(half3 Ray,half2 CellSize,half3 RayDir){//获取的是格子左下角的X坐标
    half Result=floor(Ray.x*CellSize.x+sign(RayDir.x)+(1e-3f).x*rcp(CellSize.x))/CellSize.x;
    return Result;
}
half AlignedCellYMarching(half3 Ray,half2 CellSize,half3 RayDir){//获取的是格子左下角的X坐标
    half Result=floor(Ray.y*CellSize.y+sign(RayDir.y)+(1e-3f).x*rcp(CellSize.y))/CellSize.y;
    return Result;
}
half LinearEyeDepth(half NDCDepth){
    return LinearEyeDepth(NDCDepth,_ZBufferParams);
}
void UpdateRay(inout half3 Ray,out half2 RayUv,half3 RayDir,half2 TexelSize){
    half3 Ray_Pre=Ray;
    half2 Cell_Pre=GetCurCell(Ray_Pre,TexelSize);
    
    [branch]
    if(abs(RayDir.y)<abs(RayDir.x)){
        Ray.x=AlignedCellXMarching(Ray,TexelSize,RayDir);
        Ray.y+=(Ray.x-Ray_Pre.x)*RayDir.y/RayDir.x;
        half2 Cell_Cur=GetCurCell(Ray,TexelSize);
        if(Cell_Cur.y!=Cell_Pre.y){
            Ray=Ray_Pre;
            Ray.y=AlignedCellYMarching(Ray,TexelSize,RayDir);
            Ray.x+=(Ray.y-Ray_Pre.y)*RayDir.x*rcp(RayDir.y);
        }
    }
    else{
        Ray.y=AlignedCellYMarching(Ray,TexelSize,RayDir);
        Ray.x+=(Ray.y-Ray_Pre.y)*RayDir.x/RayDir.y;
        half2 Cell_Cur=GetCurCell(Ray,TexelSize);
        if(Cell_Cur.x!=Cell_Pre.x){
            Ray=Ray_Pre;
            Ray.x=AlignedCellXMarching(Ray,TexelSize,RayDir);
            Ray.y+=(Ray.x-Ray_Pre.x)*RayDir.y*rcp(RayDir.x);
        }
    }
    RayUv=(Ray.xy+Ray_Pre.xy)*0.5f;
    Ray.z=Ray_Pre.z+RayDir.z*length(Ray.xy-Ray_Pre.xy)/length(RayDir.xy);
}
RayTraceResult RayTrace_HierarchicalZ(Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,
                            half2 uv,half3 RayOriginVs,half2 TexelSize,half2 TexelSize_HiZBuffer,half Thickness,uint RandomSeed){//TexelSize横纵像素数
    RayTraceResult Result_None;
    Result_None.SceneColor=0;
    Result_None.HitZBufferdepth=0;
    Result_None.Hituv=0;
    Result_None.HitMask=false;
    RayTraceResult Result;
    Result=Result_None;

    RayTraceResult Result_Red;
    Result_Red=Result_None;
    Result_Red.HitMask=true;
    Result_Red.SceneColor=half3(1,0,0);

    RayTraceResult Result_Green;
    Result_Green=Result_None;
    Result_Green.HitMask=true;
    Result_Green.SceneColor=half3(0,1,0);

    RayTraceResult Result_Blue;
    Result_Blue=Result_None;
    Result_Blue.HitMask=true;
    Result_Blue.SceneColor=half3(0,0,1);

    RayTraceResult Result_Pink;
    Result_Pink=Result_None;
    Result_Pink.HitMask=true;
    Result_Pink.SceneColor=half3(1,0,1);

    [flatten]
    if(RayDirVs.z>0){
        RayDirVs*=(-N-RayOriginVs.z)/RayDirVs.z;
    }

    half Random_Thickness=lerp(0.8,1.5,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed))));
    half Random_Step=lerp(0.6,1.4,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed*2))));
    half Random_Step_Bias=lerp(0.0,0.5,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed*4))));
    //Thickness*=Random_Thickness;

    half3 RayEndVs_Max=RayOriginVs+RayDirVs;
    half3 RayOriginSS=GetPositionSS(RayOriginVs).xyz;
    half3 RayEndSS=GetPositionSS(RayEndVs_Max).xyz;
    half3 RayDirSS=normalize(RayEndSS-RayOriginSS);

    half3 Ray=RayOriginSS;
    Ray.xy+=(1e-3f).xx*rcp(TexelSize_HiZBuffer.xy)*sign(RayDirSS.xy);
    half3 Ray_Pre;
    int CurMiplevel=0;

    [loop]
    for(int i=1;i<=MaxIterations;i++){
        Ray_Pre=Ray;
        half2 HitUv;
        half2 TexelSize_HiZBuffer_Cur=TexelSize_HiZBuffer.xy/(half)exp2(CurMiplevel);
        UpdateRay(Ray,HitUv,RayDirSS,TexelSize_HiZBuffer_Cur);
        [branch]
        if(Ray.x>1.0 || Ray.x<0.0 || Ray.y>1.0 || Ray.y<0.0 || Ray.z<=0.0 || Ray.z>=1.0){return Result_Red;}
        half DepthFront=HiZMipmap_WithMipmap.SampleLevel(Point_Clamp,MipmapUv_Bias(HitUv,CurMiplevel),0).r;
        half DepthBack=HiZMipmap_WithMipmap.SampleLevel(Point_Clamp,MipmapUv_Bias(HitUv,CurMiplevel),0).g;
        
        half h0=Ray_Pre.z-DepthFront;
        half h1=Ray.z-DepthFront;
        half c=abs(h0)/(abs(h1)+abs(h0));

        DepthFront=LinearEyeDepth(DepthFront);
        DepthBack=LinearEyeDepth(DepthBack);
        
        [branch]
        if(LinearEyeDepth(Ray.z)>DepthFront && LinearEyeDepth(Ray.z)<DepthFront+max(DepthBack-DepthFront,Thickness)){
            [branch]
            if(CurMiplevel!=0){
                CurMiplevel--;
                Ray=Ray_Pre;
            }
            else{
                half2 Hituv_Accurate=lerp(Ray_Pre.xy,Ray.xy,c);
                [branch]
                if(dot(normalize(RayDirVs),GetNormalVs(HitUv.xy))<0.0f){
                    Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,Hituv_Accurate.xy,0).xyz;
                    Result.Hituv=Hituv_Accurate.xy;
                    Result.HitZBufferdepth=lerp(Ray_Pre.z,Ray.z,c);
                    Result.HitMask=true;
                    return Result;
                }
                else{
                    CurMiplevel=min(CurMiplevel+1,MaxMipLevel);
                }
            }
        }
        else{
            CurMiplevel=min(CurMiplevel+1,MaxMipLevel);
        }
    }
    return Result_Pink;
}
RayTraceResult RayTrace_Linear(Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,
                            half2 uv,half3 RayOriginVs,half2 TexelSize,half Thickness,uint RandomSeed){//TexelSize横纵像素数
    RayTraceResult Result_None;
    Result_None.SceneColor=0;
    Result_None.HitZBufferdepth=0;
    Result_None.Hituv=0;
    Result_None.HitMask=false;
    RayTraceResult Result;
    Result=Result_None;

    [flatten]
    if(RayDirVs.z>0){
        RayDirVs*=(-N-RayOriginVs.z)/RayDirVs.z;
    }

    half Random_Thickness=lerp(0.8,1.5,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed))));
    half Random_Step=lerp(0.8,1.4,frac(GenerateHashedRandomFloat(uint4(uv*TexelSize,abs(_SinTime.x)*2000,RandomSeed*2))));
    Thickness*=Random_Thickness;

    half3 RayEndVs_Max=RayOriginVs+RayDirVs;
    half3 RayOriginSS=GetPositionSS(RayOriginVs).xyz;
    half3 RayEndSS=GetPositionSS(RayEndVs_Max).xyz;
    half3 RayDirSS=RayEndSS-RayOriginSS;

    half Delta;
    [flatten]
    if(abs(RayDirSS.y)>abs(RayDirSS.x)){
        RayDirSS/=abs(RayDirSS.y)*TexelSize.y;
    }
    else{
        RayDirSS/=abs(RayDirSS.x)*TexelSize.x;
    }
    RayDirSS*=Random_Step;

    half3 Ray=RayOriginSS;
    Ray.xy+=(0.5f).xx*half2(sign(RayDirSS.x),sign(RayDirSS.y))/TexelSize.xy;
    half3 Ray_Pre;

    [loop]
    for(int i=1;i<=MaxIterations;i++){
        Ray_Pre=Ray;
        Ray+=RayDirSS;
        [branch]
        if(Ray.x>1.0 || Ray.x<0.0 || Ray.y>1.0 || Ray.y<0.0){return Result_None;}
        #if defined BILINEAR_DEPTHBUFFER
        half Delta=LinearEyeDepth(Ray.z,_ZBufferParams)-SampleEyeDepthBilinear(Ray.xy);
        #else
        half Delta=LinearEyeDepth(Ray.z,_ZBufferParams)-LinearEyeDepth(SampleSceneDepth(Ray.xy),_ZBufferParams);
        #endif

        #if defined BINARY_SEARCH
        if(Delta>0 && Delta<Thickness*BinarySearch_Tolerance){
            half3 RayStart_Bin=Ray_Pre;
            half3 RayEnd_Bin=Ray;
            RayDirSS*=0.5f;
            half3 RayMid=RayStart_Bin+RayDirSS;
            half Delta_Bin;
            [loop]
            for(int j=0;j<1+BINARYSEARCHITERATIONS;j++){
                RayDirSS*=0.5f;
                #if defined BILINEAR_DEPTHBUFFER
                Delta_Bin=LinearEyeDepth(RayMid.z,_ZBufferParams)-SampleEyeDepthBilinear(RayMid.xy);
                #else
                Delta_Bin=LinearEyeDepth(RayMid.z,_ZBufferParams)-LinearEyeDepth(SampleSceneDepth(RayMid.xy),_ZBufferParams);
                #endif
                [branch]
                if(Delta_Bin>0.0f){//超过
                    RayMid-=RayDirSS;
                }
                else{//没超过
                    RayMid+=RayDirSS;
                }
            }
            [branch]
            if(dot(normalize(RayDirVs),GetNormalVs(RayMid.xy))>0.0){return Result_None;}
            half d=LinearEyeDepth(Ray.z,_ZBufferParams)-LinearEyeDepth(SampleSceneDepth(Ray.xy),_ZBufferParams);
            d/=Thickness;
            Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,RayMid.xy,0).xyz*(1-pow8(abs(Delta_Bin/Thickness)));
            Result.Hituv=RayMid.xy;
            Result.HitZBufferdepth=RayMid.z;
            Result.HitMask=true;
            return Result;
        }
        #else
        [branch]
        if(Delta>0 && Delta<Thickness){
            [branch]
            if(dot(normalize(RayDirVs),GetNormalVs(Ray.xy))>0.0){return Result_None;}
            Result.SceneColor=SAMPLE_TEXTURE2D_LOD(SceneColor,sampler_SceneColor,Ray.xy,0).xyz*(1-pow8(abs(Delta/Thickness)));
            Result.Hituv=Ray.xy;
            Result.HitZBufferdepth=Ray.z;
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

    [branch]
    if(GetSkyBoxMask(uv)){return Result_None;}
    
    half Roughness=1-SAMPLE_TEXTURE2D(_GBuffer2,sampler_GBuffer2,uv).a;
    [branch]
    if(Roughness>MAXROUGHNESS){return Result_None;}
    half3 RayStart=GetPositionVs_NoFilter(uv);
    half3 ViewDir=GetViewDir(uv);
    half3 NormalVs=GetNormalVs(uv);

    // [branch]
    // if(NormalVs.z<0){return Result_None;}

    half2 TexelSize=RT_TexelSize.zw;

    half3 result=0;
    [branch]
    if(Roughness>=MirrorReflectionThreshold){
        half3x3 Tan2View=GetTanToViewMatrix(NormalVs);
        half3x3 View2Tan=GetViewToTanMatrix(NormalVs);
        bool Any_Hit=false;
        int Count_Hit=0;

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

        #if defined HIZ_ACCELERATE
        RayTraceResult RayTraceResult_one=RayTrace_HierarchicalZ(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,RT_TexelSize_HiZBuffer,THICKNESS,2);
        #else
        RayTraceResult RayTraceResult_one=RayTrace_Linear(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,THICKNESS,2);
        #endif
        Result.SceneColor_Pdf.xyz=RayTraceResult_one.SceneColor;
        Result.HitPoint_Mask.xyz=half3(RayTraceResult_one.Hituv,RayTraceResult_one.HitZBufferdepth);
        Result.SceneColor_Pdf.w=Pdf_tmp;
        Result.HitPoint_Mask.w=RayTraceResult_one.HitMask;
    }
    else{
        half3 RayDir=GetReflectDir(ViewDir,NormalVs);
        #if defined HIZ_ACCELERATE
        RayTraceResult RayTraceResult_one=RayTrace_HierarchicalZ(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,RT_TexelSize_HiZBuffer,THICKNESS,1);
        #else
        RayTraceResult RayTraceResult_one=RayTrace_Linear(SSR_CameraTexture,Linear_Clamp,ViewDir,RayDir,uv,RayStart,TexelSize,THICKNESS,1);
        #endif
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

    [branch]
    if(GetSkyBoxMask(uv)){return 0.0;}
    half3 positionVs=GetPositionVs_NoFilter(uv);
    half3 ViewDir=GetViewDir(uv);
    half3 NormalVs=GetNormalVs(uv);
    half Roughness=1-SAMPLE_TEXTURE2D(_GBuffer2,sampler_GBuffer2,uv).a;
    [branch]
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
        [branch]
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
    [branch]
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
    [branch]
    if (r.x > rmax.x + eps)
        r *= (rmax.x / r.x);
    [branch]
    if (r.y > rmax.y + eps)
        r *= (rmax.y / r.y);
    [branch]
    if (r.z > rmax.z + eps)
        r *= (rmax.z / r.z);
    [branch]
    if (r.x < rmin.x - eps)
        r *= (rmin.x / r.x);
    [branch]
    if (r.y < rmin.y - eps)
        r *= (rmin.y / r.y);
    [branch]
    if (r.z < rmin.z - eps)
        r *= (rmin.z / r.z);

    return AABB_Avg + r;
}
TEXTURE2D(aaa);
half4 Frag_StochasticSSR_TemporalFilter(VertexOutput i):SV_Target{
    half2 uv=i.uv;

    [branch]
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