Shader "PostProcess/HiZMipmap"
{
    SubShader
    {
        Pass
        {
        Name "HiZ_Level_0"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "HiZMipmap.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_HiZMip_0
            ENDHLSL
        }
        Pass
        {
        Name "HiZ_Level_Other"
        ZWrite Off
        ZTest Always
        CUll off
        Tags{"LightMode"="PostProcess"}
            HLSLPROGRAM
            #include "HiZMipmap.hlsl"
            #pragma vertex Vert_PostProcessDefault
            #pragma fragment Frag_HiZMip_Other
            ENDHLSL
        }
    }
}