#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
#include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/DeclareDepthTexture.hlsl"

float2 HizDestTexelSize;
TEXTURE2D_X_FLOAT(RT_HizSourceTex_In);
SamplerState point_clamp;
TEXTURE2D_X_FLOAT(HiZMipmap_Level_0);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_1);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_2);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_3);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_4);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_5);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_6);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_7);
TEXTURE2D_X_FLOAT(HiZMipmap_Level_8);
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
float2 MipmapUv_Bias(float2 uv,float4 Uv_Bias_Lut){
    uv=saturate(uv);
    return float2(lerp(Uv_Bias_Lut.x,Uv_Bias_Lut.z,uv.x),lerp(Uv_Bias_Lut.y,Uv_Bias_Lut.w,uv.y));
}
float2 MipmapUv_Bias_Inv(float2 uv,float4 Uv_Bias_Lut_Inv,out bool flag){
    float2 Result;
    Result.x=Uv_Bias_Lut_Inv.x*uv.x-Uv_Bias_Lut_Inv.z;
    Result.y=Uv_Bias_Lut_Inv.y*uv.y-Uv_Bias_Lut_Inv.w;
    if(Result.x>1 || Result.x<0 || Result.y>1 || Result.y<0){flag=false;}
    else{flag=true;}
    return Result;
}

float2 Frag_HiZMip_0(VertexOutput i):SV_Target{
    return SampleSceneDepth(i.uv).xx;
}
float2 Frag_HiZMip_Other(VertexOutput i):SV_Target{
    float2 uv=i.uv;
    uv=floor(uv*HizDestTexelSize)/HizDestTexelSize;

    float2 HizSourceTexelSize=HizDestTexelSize*2.0;
    HizSourceTexelSize=rcp(HizSourceTexelSize);
    HizDestTexelSize=rcp(HizDestTexelSize);

    float2 du=float2(HizSourceTexelSize.x,0.0);
    float2 dv=float2(0.0,HizSourceTexelSize.y);
    float2 Depth_00=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv,0).rg;
    float2 Depth_01=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+dv,0).rg;
    float2 Depth_10=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+du,0).rg;
    float2 Depth_11=SAMPLE_TEXTURE2D_X_LOD(RT_HizSourceTex_In,point_clamp,uv+dv+du,0).rg;
    float Depth_Max=max(Depth_11.x,max(Depth_10.x,max(Depth_00.x,Depth_01.x)));
    float Depth_Min=min(Depth_11.y,min(Depth_10.y,min(Depth_00.y,Depth_01.y)));
    return float2(Depth_Max,Depth_Min);
}
float2 Frag_HiZMip_CombineMipmap(VertexOutput i):SV_Target{
    float4 Uv_Bias_Lut[9]={float4(0.0f,0.333333334f,1.0f,1.0f),
                            float4(0.0f,0.0f,0.5f,0.333333333f),
                            float4(0.5f,0.16666667f,0.75f,0.333333333f),
                            float4(0.75f,0.25f,0.875f,0.333333333f),
                            float4(0.875f,0.2916666667f,0.9375f,0.333333333f),
                            float4(0.9375f,0.3125f,0.96875f,0.333333333f),
                            float4(0.96875f,0.3229166666667f,0.984375f,0.333333333f),
                            float4(0.984375f,0.328125f,0.9921875f,0.333333333f),
                            float4(0.9921875f,0.33072916666667f,0.99609375f,0.333333333f)};
    float4 Uv_Bias_Inv_Lut[9]={float4(1.0f,1.5f,0.0f,0.5f),
                            float4(2.0f,3.0f,0.0f,0.0f),
                            float4(4.0f,6.0f,2.0f,1.0f),
                            float4(8.0f,12.0f,6.0f,3.0f),
                            float4(16.0,24.0,14.0f,7.0f),
                            float4(32.0f,48.0f,30.0f,15.0f),
                            float4(64.0f,96.0f,62.0f,31.0f),
                            float4(128.0f,192.0f,126.0f,63.0f),
                            float4(256.0f,384.0f,254.0f,127.0f)};
    float2 uv=i.uv;
    bool flag=false;
    float2 Result=0.0f.xx;
    float2 Temp=0.0f.xx;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_0,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[0],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_1,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[1],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_2,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[2],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_3,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[3],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_4,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[4],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_5,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[5],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_6,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[6],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_7,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[7],flag),0).rg;
    Result+=Temp*flag;
    Temp=SAMPLE_TEXTURE2D_X_LOD(HiZMipmap_Level_8,point_clamp,MipmapUv_Bias_Inv(uv,Uv_Bias_Inv_Lut[8],flag),0).rg;
    Result+=Temp*flag;
    return Result;
}