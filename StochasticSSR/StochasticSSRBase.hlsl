#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareNormalsTexture.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/UnityGBuffer.hlsl"

#if defined FULL_PRECISION_SSR
#define half float
#define half2 float2
#define half3 float3
#define half4 float4
#define half4x4 float4x4
#define half3x3 float3x3
#endif

TEXTURE2D_X_FLOAT(_CameraDepthTexture);
SamplerState Point_Clamp;
SamplerState Linear_Clamp;
float SampleSceneDepth(float2 uv)
{
    return SAMPLE_TEXTURE2D_X(_CameraDepthTexture, Point_Clamp, UnityStereoTransformScreenSpaceTex(uv)).r;
}

half4 test_TexelSize;
TEXTURE2D(test);

#define TexelSize_Bilinear SSR_CameraTexture_TexelSize.zw

half4x4 World2View_Matrix;
half4x4 View2World_Matrix;
half4x4 InvProjection_Matrix;
half4x4 Projection_Matrix;
half fov;
half F,N;

half Bilinear_Tolerance;
half BinarySearch_Tolerance;
half THICKNESS;
half STEPLENGTH;
int SAMPLESPERPIXEL;

TEXTURE2D(SSR_CameraTexture);
SAMPLER(sampler_SSR_CameraTexture);

TEXTURE2D(_GBuffer0);
SAMPLER(sampler_GBuffer0);
TEXTURE2D(_GBuffer1);
SAMPLER(sampler_GBuffer1);
TEXTURE2D(_GBuffer2);
SAMPLER(sampler_GBuffer2);

half4 SSR_CameraTexture_TexelSize;

half EyeDepthToZbuffer(half eyedepth){
    return ((1/eyedepth)-_ZBufferParams.w)/_ZBufferParams.z;
}
half SampleEyeDepthBilinear(half2 uv){
    half2 uv_raw=uv;

    uv+=half2(-0.5,-0.5)/TexelSize_Bilinear;
    half2 uv_frac=frac(uv*TexelSize_Bilinear);
    half2 uv_00,uv_01,uv_10,uv_11;
    uv_00=uv;
    uv_01=uv_00+half2(0,1)/TexelSize_Bilinear;
    uv_10=uv_00+half2(1,0)/TexelSize_Bilinear;
    uv_11=uv_00+half2(1,1)/TexelSize_Bilinear;

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
half Noise2D(half2 p)
{
    p*=1000.0;
    p*=1+_SinTime.y*0.5;
    half3 p3  = frac(half3(p.xyx) * .1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return saturate(frac((p3.x + p3.y) * p3.z));
}
half Pow2(half x){
    return x*x;
}
half Pow64(half x){
    half Out=Pow2(x);
    Out=Pow2(Out);
    Out=Pow2(Out);
    Out=Pow2(Out);
    Out=Pow2(Out);
    Out=Pow2(Out);
    Out=Pow2(Out);
    Out=Pow2(Out);
    return Out;
}
half3 GetReflectDir(half3 ViewDir,half3 normalVS){
    half cos=dot(ViewDir,normalVS);
    return normalize(2*normalVS*cos-ViewDir);
}
bool GetSkyBoxMask(half2 uv){
    return SAMPLE_TEXTURE2D(_GBuffer0,sampler_GBuffer0,uv).w;
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
    return -normalize(Norm)*step(GetEyeDepth(uv),9986);
}
half PerspectiveCorrectInterpolateCoefficient_SS(half t,half z0,half z1){//z=z1*t+z0*(1-t)
    return -(t*z1)/(z1*t+(1-t)*z0);
}
half PerspectiveCorrectInterpolateCoefficient_VS(half s,half z0,half z1){//z=z1*t+z0*(1-t)
    return (z0*s)/(s*z0-z1*(1+s));
}

half4 RayMarching(Texture2D SceneColor,SamplerState sampler_SceneColor,half3 ViewDir,half3 RayDirVs,half2 uv,half3 RayOriginVs,half2 TexelSize,half StepLength,half Thickness,half2 RandomSeed){//TexelSize横纵像素数
    half Vertical_Extension;
    half Vertical_Extension_Plane;
    UNITY_BRANCH
    if(RayDirVs.z>0){
        Vertical_Extension=(-N-RayOriginVs.z)/RayDirVs.z;
        RayDirVs*=Vertical_Extension;
    }

    half Random_Thickness=lerp(0.6,1.4,Noise2D(uv*RandomSeed+RandomSeed));
    Thickness*=Random_Thickness;

    half3 RayEndVs_Max=RayOriginVs+RayDirVs;
    half2 RayOriginSS=GetPositionSS(RayOriginVs).xy;
    half2 RayEndSS=GetPositionSS(RayEndVs_Max).xy;
    half2 RayDirSS=RayEndSS-RayOriginSS;

    half L=length(RayDirSS);
    half MinLengthSS=length(rcp(TexelSize.xy));
    half Random_Step=lerp(0.6,1.4,Noise2D(uv*RandomSeed*1.5f+RandomSeed));
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
        if(Ray.x>1.0 || Ray.x<0.0 || Ray.y>1.0 || Ray.y<0.0){return 0.0;}
        
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
            for(int j=0;j<10;j++){
                MidZ=lerp(StartZ,EndZ,PerspectiveCorrectInterpolateCoefficient_VS(0.5,StartZ,EndZ));
                MidRay=0.5*(RayStartSS_Bin+RayEndSS_Bin);
                if(MidRay.x>1.0 || MidRay.x<0.0 || MidRay.y>1.0 || MidRay.y<0.0){return 0.0;}
                Delta_Bin=-MidZ-GetEyeDepth(MidRay);
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
            if(dot(normalize(RayDirVs),GetNormalVs(MidRay.xy))<0.0){return 0.0;}
            UNITY_BRANCH
            if(Delta_Bin<Thickness){
                return half4(SAMPLE_TEXTURE2D(SceneColor,sampler_SceneColor,MidRay.xy).xyz,1);
            }
            
        }
        #else
        UNITY_BRANCH
        if(Delta>0 && Delta<Thickness){
            UNITY_BRANCH
            if(dot(normalize(RayDirVs),GetNormalVs(Ray.xy))<0.0){return 0.0;}
            return half4(SAMPLE_TEXTURE2D(SceneColor,sampler_SceneColor,Ray.xy).xyz,1);
        }
        #endif
    }
    return 0;
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
half4 Frag_StochasticSSR(VertexOutput i):SV_Target{
    half2 uv=i.uv;

    if(GetSkyBoxMask(uv)){return 0.0;}
    
    half3 RayStart=GetPositionVs(uv);
    half3 ViewDir=GetViewDir(uv);
    half3 NormalVs=GetNormalVs(uv);
    half3 RayDir=GetReflectDir(ViewDir,NormalVs);
    half2 TexelSize=SSR_CameraTexture_TexelSize.zw;

    half3 result=0;
    int Count_Hit=0;
    UNITY_LOOP
    for(int i=1;i<=SAMPLESPERPIXEL;i++){
        half4 result_one=RayMarching(SSR_CameraTexture,sampler_SSR_CameraTexture,ViewDir,RayDir,uv,RayStart,TexelSize,STEPLENGTH,THICKNESS,half2(i,i));
        UNITY_BRANCH
        if(result_one.w==1.0){
            Count_Hit++;
        }
        result+=result_one.xyz;
    }
    result/=Count_Hit;
    return half4(result.xyz,1);
}