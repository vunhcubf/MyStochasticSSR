Shader "PostProcess/StochasticSSR"
{
    SubShader
    {
        Pass
        {
        Name "StochasticSSR"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "StochasticSSRBase.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_StochasticSSR
            
            #pragma shader_feature _GBUFFER_NORMALS_OCT
            #pragma shader_feature BINARY_SEARCH
            #pragma shader_feature BILINEAR_DEPTHBUFFER
            #pragma shader_feature FULL_PRECISION_SSR
            ENDHLSL
        }
    }
}