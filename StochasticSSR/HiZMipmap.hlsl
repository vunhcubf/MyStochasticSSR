#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

float2 HizDestTexelSize;
TEXTURE2D_X_FLOAT(RT_HizSourceTex_In);
SamplerState point_clamp;
struct VertexInput{
    float4 positionOS:POSITION;
    float2 uv:TEXCOORD0;
};
struct VertexOutput{
    float4 position:SV_POSITION;
    float2 uv:TEXCOORD0;
};
VertexOutput Vert_PostProcessDefault(VertexInput v)
{
    VertexOutput o;
    VertexPositionInputs positionInputs = GetVertexPositionInputs(v.positionOS.xyz);
    o.position = positionInputs.positionCS;
    o.uv = v.uv;
    return o;
}
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

float Frag_HiZMip_0(VertexOutput i):SV_Target{
    return SampleSceneDepth(i.uv);
}
float Frag_HiZMip_Other(VertexOutput i):SV_Target{
    float2 uv=i.uv;
    uv=floor(uv*HizDestTexelSize)/HizDestTexelSize;

    float2 HizSourceTexelSize=HizDestTexelSize*2.0;
    HizSourceTexelSize=rcp(HizSourceTexelSize);
    HizDestTexelSize=rcp(HizDestTexelSize);

    float2 du=float2(HizSourceTexelSize.x,0.0);
    float2 dv=float2(0.0,HizSourceTexelSize.y);
    float Depth_00=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv,0).r;
    float Depth_01=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+dv,0).r;
    float Depth_10=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+du,0).r;
    float Depth_11=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+dv+du,0).r;
    float Depth_Max=max(Depth_11,max(Depth_10,max(Depth_00,Depth_01)));
    return Depth_Max;
}