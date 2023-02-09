Shader "PostProcess/StochasticSSR"
{
    SubShader
    {
        Pass
        {
        Name "StochasticSSR_RayTrace"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "StochasticSSRBase.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_StochasticSSR_RayTrace
            
            #pragma shader_feature _GBUFFER_NORMALS_OCT
            #pragma shader_feature BINARY_SEARCH
            #pragma shader_feature BILINEAR_DEPTHBUFFER
            #pragma shader_feature FULL_PRECISION_SSR
            #pragma shader_feature HIZ_ACCELERATE
            ENDHLSL
        }
        Pass
        {
        Name "StochasticSSR_Resolve"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "StochasticSSRBase.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_StochasticSSR_Resolve
            
            #pragma shader_feature _GBUFFER_NORMALS_OCT
            #pragma shader_feature FULL_PRECISION_SSR
            ENDHLSL
        }
        Pass
        {
        Name "StochasticSSR_Temporal"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "StochasticSSRBase.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_StochasticSSR_TemporalFilter
            
            #pragma shader_feature _GBUFFER_NORMALS_OCT
            #pragma shader_feature FULL_PRECISION_SSR
            ENDHLSL
        }
    }
}