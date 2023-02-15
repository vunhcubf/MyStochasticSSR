using System;
using System.Collections.Generic;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
using static Unity.Burst.Intrinsics.X86.Avx;
using ProfilingScope = UnityEngine.Rendering.ProfilingScope;
//后处理可以用这个
namespace UnityEngine.Rendering.Universal
{
    [DisallowMultipleRendererFeature]
    public class StochasticSSR : ScriptableRendererFeature
    {
        [System.Serializable]
        public class Setting
        {
            public string RenderPassName = "StochasticSSR";
            public RenderPassEvent passEvent = RenderPassEvent.BeforeRenderingPostProcessing;
            public int passEventOffset = 2;
            public bool FullPrecision = false;
            public bool Debug = false;
            [Range(0.0f, 4.0f)]
            public float RayBias = 0.5f;
            public Texture BlueNoise;

            [Space(10)]
            [Header("HiZ加速结构")]
            [InspectorToggleLeft] public bool HierarchicalZBuffer;
            [Range(1, 8)] public int HiZLevels = 4;
            [Range(0.0f, 1.0f)] public double MipmapBias_Width = 1.0f;
            [Range(0.0f, 1.0f)] public double MipmapBias_Height = 1.0f;

            [Space(10)]
            [Header("二分查找")]
            [InspectorToggleLeft] public bool BinarySearch = true;
            [Range(1.0f, 3.0f)] public float BinarySearch_Tolerance = 1.4f;
            public int BinarySearchIterations = 5;

            [Space(10)]
            [Header("深度图线性过滤")]
            [InspectorToggleLeft] public bool BilinearDepthBuffer = true;
            public float Bilinear_Tolerance = 0.5f;

            [Space(10)]
            [Header("SSR设置")]
            public float Thickness = 0.1f;
            public int MaxIterations = 100;
            public float AngleBias = 0.1f;
            public float MaxRoughness = 0.5f;
            public float MirrorReflectionThreshold = 0.05f;

            [Space(10)]
            [Header("降噪设置")]
            [InspectorToggleLeft] public bool TemporalFilter = true;
            [Range(0f, 1f)] public float TemporalFilterIntensity = 0.2f;
        }
        public Setting Settings = new Setting();
        class RenderPass : ScriptableRenderPass
        {

            private static readonly int P_World2View_Matrix_ID = Shader.PropertyToID("World2View_Matrix");
            private static readonly int P_View2World_Matrix_ID = Shader.PropertyToID("View2World_Matrix");
            private static readonly int P_InvProjection_Matrix_ID = Shader.PropertyToID("InvProjection_Matrix");
            private static readonly int P_Projection_Matrix_ID = Shader.PropertyToID("Projection_Matrix");
            private static readonly int P_fov_ID = Shader.PropertyToID("fov");
            private static readonly int P_Aspect_ID = Shader.PropertyToID("Aspect");
            private static readonly int P_FarClipPlane_ID = Shader.PropertyToID("F");
            private static readonly int P_NearClipPlane_ID = Shader.PropertyToID("N");
            private static readonly int P_Bilinear_Tolerance_ID = Shader.PropertyToID("Bilinear_Tolerance");
            private static readonly int P_BinarySearch_Tolerance_ID = Shader.PropertyToID("BinarySearch_Tolerance");
            private static readonly int P_Thickness_ID = Shader.PropertyToID("THICKNESS");
            private static readonly int P_MaxIterations_ID = Shader.PropertyToID("MaxIterations");
            private static readonly int P_AngleBias_ID = Shader.PropertyToID("ANGLE_BIAS");
            private static readonly int P_MaxRoughness_ID = Shader.PropertyToID("MAXROUGHNESS");
            private static readonly int P_BinarySearchIterations_ID = Shader.PropertyToID("BINARYSEARCHITERATIONS");
            private static readonly int P_ScreenParams_ID = Shader.PropertyToID("RT_TexelSize");
            private static readonly int P_HiZBufferTexelSize_ID = Shader.PropertyToID("RT_TexelSize_HiZBuffer");
            private static readonly int P_MirrorReflectionThreshold_ID = Shader.PropertyToID("MirrorReflectionThreshold");
            private static readonly int P_TemporalFilterIntensity_ID = Shader.PropertyToID("TemporalFilterIntensity");
            private static readonly int P_RayBias_ID = Shader.PropertyToID("RayBias");
            private static readonly int P_BlueNoise_ID = Shader.PropertyToID("BlueNoise");
            private static readonly int P_BlueNoise_TexelSize_ID = Shader.PropertyToID("BlueNoise_TexelSize");

            private static readonly int P_MaxMipLevels_ID = Shader.PropertyToID("MaxMipLevel");

            private static readonly int RT_SSR_CameraTexture_ID = Shader.PropertyToID("SSR_CameraTexture");
            private static readonly int RT_SSR_Result1_ID = Shader.PropertyToID("RT_SSR_Result1");
            private static readonly int RT_SSR_Result2_ID = Shader.PropertyToID("RT_SSR_Result2");
            private static readonly int RT_SSR_Depth_None_ID = Shader.PropertyToID("RT_None");
            private static readonly int RT_SSR_Resolved_ID = Shader.PropertyToID("RT_SSR_Resolve");
            private static readonly int RT_SSR_TemporalFilter_In_ID = Shader.PropertyToID("RT_SSR_TemporalFilter_In");
            private static readonly int RT_SSR_Denoised_ID = Shader.PropertyToID("RT_SSR_Denoised");
            private static readonly int RT_SSR_Denoised_Pre_ID = Shader.PropertyToID("RT_SSR_Denoised_Pre");
            private static readonly int GLOBAL_RT_Stochastic_SSR_ID = Shader.PropertyToID("Stochastic_SSR");

            private static readonly int RT_HiZMipmap_WithMipmap = Shader.PropertyToID("HiZMipmap_WithMipmap");
            private static readonly int[] RT_HiZMipmap_Levels_ID = new int[9] {
        Shader.PropertyToID("HiZMipmap_Level_0"),
        Shader.PropertyToID("HiZMipmap_Level_1"),
        Shader.PropertyToID("HiZMipmap_Level_2"),
        Shader.PropertyToID("HiZMipmap_Level_3"),
        Shader.PropertyToID("HiZMipmap_Level_4"),
        Shader.PropertyToID("HiZMipmap_Level_5"),
        Shader.PropertyToID("HiZMipmap_Level_6"),
        Shader.PropertyToID("HiZMipmap_Level_7"),
        Shader.PropertyToID("HiZMipmap_Level_8") };

            private static readonly int RT_Mip_In_ID = Shader.PropertyToID("RT_HizSourceTex_In");
            private static readonly int P_HizDestTexelSize_ID = Shader.PropertyToID("HizDestTexelSize");

            private Setting Settings;
            private Material SSRMaterial;
            private RenderTexture RT_SSR_Denoised_Pre;

            private int[] HiZMipmapBuffers_ID;
            private int[] MipmapWidth;//0到max,max+1个
            private int[] MipmapHeight;
            private int MaxMipLevel = 0;
            private void InitializeHiZMipmap(int HiZLevels, RenderingData renderingData, double MipmapBias_Height, double MipmapBias_Width, CommandBuffer cmd)
            {
                float RenderScale = renderingData.cameraData.renderScale;
#if UNITY_EDITOR
                RenderScale = 1.0f;
#endif
                Camera camera = renderingData.cameraData.camera;
                int PixelWidth = (int)(RenderScale * (float)camera.pixelWidth);
                int PixelHeight = (int)(RenderScale * (float)camera.pixelHeight);

                int Mip0Width = (int)Math.Round(Math.Pow(2.0, Math.Floor(Math.Log(PixelWidth, 2.0) + MipmapBias_Width)));
                int MaxMipMap_Width = (int)Math.Log(Mip0Width, 2.0);
                int Mip0Height = (int)Math.Round(Math.Pow(2.0, Math.Floor(Math.Log(PixelHeight, 2.0) + MipmapBias_Height)));
                int MaxMipMap_Height = (int)Math.Log(Mip0Height, 2.0);
                MaxMipLevel = Math.Min(MaxMipMap_Width, MaxMipMap_Height);
                MaxMipLevel = Math.Min(MaxMipLevel, HiZLevels);

                MipmapWidth = new int[MaxMipLevel + 1];
                MipmapHeight = new int[MaxMipLevel + 1];
                HiZMipmapBuffers_ID = new int[MaxMipLevel + 1];

                MipmapHeight[0] = Mip0Height;
                MipmapWidth[0] = Mip0Width;
                HiZMipmapBuffers_ID[0] = Shader.PropertyToID("HiZMipmapBuffers_" + 0);
                cmd.GetTemporaryRT(HiZMipmapBuffers_ID[0], MipmapWidth[0], MipmapHeight[0], 0, FilterMode.Point, RenderTextureFormat.RGHalf);
                cmd.GetTemporaryRT(RT_HiZMipmap_WithMipmap, MipmapWidth[0], MipmapHeight[0] * 3 / 2, 0, FilterMode.Point, RenderTextureFormat.RGHalf);

                for (int i = 1; i <= MaxMipLevel; i++)
                {
                    MipmapHeight[i] = MipmapHeight[i - 1] / 2;
                    MipmapWidth[i] = MipmapWidth[i - 1] / 2;
                    HiZMipmapBuffers_ID[i] = Shader.PropertyToID("HiZMipmapBuffers_" + i);
                    cmd.GetTemporaryRT(HiZMipmapBuffers_ID[i], MipmapWidth[i], MipmapHeight[i], 0, FilterMode.Point, RenderTextureFormat.RGHalf);
                }
            }
            private void GenerateHiZBuffers(CommandBuffer cmd)
            {
                var HiZMaterial = new Material(Shader.Find("PostProcess/HiZMipmap"));

                //第0级mipmap
                cmd.SetGlobalVector(P_HizDestTexelSize_ID, new Vector2(MipmapWidth[0], MipmapHeight[0]));
                cmd.SetRenderTarget(HiZMipmapBuffers_ID[0]);
                cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, HiZMaterial, 0, 0);
                for (int i = 1; i <= MaxMipLevel; i++)
                {
                    cmd.SetGlobalVector(P_HizDestTexelSize_ID, new Vector2(MipmapWidth[i], MipmapHeight[i]));
                    cmd.SetRenderTarget(HiZMipmapBuffers_ID[i]);
                    cmd.SetGlobalTexture(RT_Mip_In_ID, HiZMipmapBuffers_ID[i - 1]);
                    cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, HiZMaterial, 0, 1);
                }
                for (int i = 0; i <= MaxMipLevel; i++)
                {
                    cmd.SetGlobalTexture(RT_HiZMipmap_Levels_ID[i], HiZMipmapBuffers_ID[i]);
                }
                SSRMaterial.SetInt(P_MaxMipLevels_ID, MaxMipLevel);
                cmd.SetRenderTarget(RT_HiZMipmap_WithMipmap);
                cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, HiZMaterial, 0, 2);
                cmd.SetGlobalTexture(RT_HiZMipmap_WithMipmap, RT_HiZMipmap_WithMipmap);
            }
            public void DestroyHiZMipmap(CommandBuffer cmd)
            {
                foreach (var A in HiZMipmapBuffers_ID)
                {
                    cmd.ReleaseTemporaryRT(A);
                }
                cmd.ReleaseTemporaryRT(RT_HiZMipmap_WithMipmap);
            }

            public RenderPass(Setting Settings)
            {
                this.Settings = Settings;
            }
            public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
            {
                //声明使用Normal
                ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Normal | ScriptableRenderPassInput.Motion);
            }
            public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
            {
                float RenderScale = renderingData.cameraData.renderScale;
#if UNITY_EDITOR
                RenderScale = 1.0f;
#endif
                Camera camera = renderingData.cameraData.camera;
                int PixelWidth = (int)(RenderScale * (float)camera.pixelWidth);
                int PixelHeight = (int)(RenderScale * (float)camera.pixelHeight);

                if (RT_SSR_Denoised_Pre is null || RT_SSR_Denoised_Pre.IsDestroyed())
                {
                    RT_SSR_Denoised_Pre = new RenderTexture(PixelWidth, PixelHeight, 0, RenderTextureFormat.DefaultHDR);
                    RT_SSR_Denoised_Pre.filterMode = FilterMode.Bilinear;
                    RT_SSR_Denoised_Pre.Create();
                }
                if (PixelWidth != RT_SSR_Denoised_Pre.width || PixelHeight != RT_SSR_Denoised_Pre.height)
                {
                    RT_SSR_Denoised_Pre.DiscardContents();
                    RT_SSR_Denoised_Pre.Release();
                    DestroyImmediate(RT_SSR_Denoised_Pre);
                    RT_SSR_Denoised_Pre = null;

                    RT_SSR_Denoised_Pre = new RenderTexture(PixelWidth, PixelHeight, 0, RenderTextureFormat.DefaultHDR);
                    RT_SSR_Denoised_Pre.filterMode = FilterMode.Bilinear;
                    RT_SSR_Denoised_Pre.Create();
                }
            }
            public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
            {
                CommandBuffer cmd = CommandBufferPool.Get(Settings.RenderPassName);
                float RenderScale = renderingData.cameraData.renderScale;
#if UNITY_EDITOR
                    RenderScale = 1.0f;
#endif
                Camera camera = renderingData.cameraData.camera;
                int PixelWidth = (int)(RenderScale * (float)camera.pixelWidth);
                int PixelHeight = (int)(RenderScale * (float)camera.pixelHeight);

                InitializeHiZMipmap(Settings.HiZLevels, renderingData, Settings.MipmapBias_Height, Settings.MipmapBias_Width, cmd);

                SSRMaterial = new Material(Shader.Find("PostProcess/StochasticSSR"));
                SSRMaterial.SetVector(P_ScreenParams_ID, new Vector4(1 / PixelWidth, 1 / PixelHeight, PixelWidth, PixelHeight));
                SSRMaterial.SetVector(P_HiZBufferTexelSize_ID, new Vector2(MipmapWidth[0], MipmapHeight[0]));

                SSRMaterial.SetMatrix(P_World2View_Matrix_ID, camera.worldToCameraMatrix);
                SSRMaterial.SetMatrix(P_View2World_Matrix_ID, camera.cameraToWorldMatrix);
                SSRMaterial.SetMatrix(P_Projection_Matrix_ID, camera.projectionMatrix);//这个是opengl的矩阵，不适用dx
                SSRMaterial.SetMatrix(P_InvProjection_Matrix_ID, camera.projectionMatrix.inverse);//这个是opengl的矩阵，不适用dx
                SSRMaterial.SetFloat(P_fov_ID, camera.fieldOfView);
                SSRMaterial.SetFloat(P_Aspect_ID, camera.aspect);
                SSRMaterial.SetFloat(P_NearClipPlane_ID, camera.nearClipPlane);
                SSRMaterial.SetFloat(P_FarClipPlane_ID, camera.farClipPlane);
                SSRMaterial.SetFloat(P_RayBias_ID, Settings.RayBias);

                if (Settings.BinarySearch) { SSRMaterial.EnableKeyword("BINARY_SEARCH"); }
                if (Settings.BilinearDepthBuffer) { SSRMaterial.EnableKeyword("BILINEAR_DEPTHBUFFER"); }
                if (Settings.FullPrecision) { SSRMaterial.EnableKeyword("FULL_PRECISION_SSR"); }
                if (Settings.HierarchicalZBuffer) { SSRMaterial.EnableKeyword("HIZ_ACCELERATE"); }

                SSRMaterial.SetFloat(P_Bilinear_Tolerance_ID, Settings.Bilinear_Tolerance);
                SSRMaterial.SetFloat(P_BinarySearch_Tolerance_ID, Settings.BinarySearch_Tolerance);
                SSRMaterial.SetFloat(P_Thickness_ID, Settings.Thickness);
                SSRMaterial.SetFloat(P_MaxIterations_ID, Settings.MaxIterations);
                SSRMaterial.SetFloat(P_AngleBias_ID, Settings.AngleBias);
                SSRMaterial.SetFloat(P_MaxRoughness_ID, Settings.MaxRoughness);
                SSRMaterial.SetInt(P_BinarySearchIterations_ID, Settings.BinarySearchIterations);
                SSRMaterial.SetFloat(P_MirrorReflectionThreshold_ID, Settings.MirrorReflectionThreshold);
                SSRMaterial.SetFloat(P_TemporalFilterIntensity_ID, Settings.TemporalFilterIntensity);
                SSRMaterial.SetTexture(P_BlueNoise_ID,Settings.BlueNoise);
                SSRMaterial.SetVector(P_BlueNoise_TexelSize_ID, new Vector2(Settings.BlueNoise.width, Settings.BlueNoise.height));

                cmd.GetTemporaryRT(RT_SSR_Result1_ID, PixelWidth, PixelHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                cmd.GetTemporaryRT(RT_SSR_Result2_ID, PixelWidth, PixelHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
                cmd.GetTemporaryRT(RT_SSR_Depth_None_ID, PixelWidth, PixelHeight, 8, FilterMode.Point, RenderTextureFormat.Depth);
                cmd.GetTemporaryRT(RT_SSR_Resolved_ID, PixelWidth, PixelHeight, 0, FilterMode.Point, RenderTextureFormat.DefaultHDR);
                cmd.GetTemporaryRT(RT_SSR_Denoised_ID, PixelWidth, PixelHeight, 0, FilterMode.Point, RenderTextureFormat.DefaultHDR);

                var CameraColorTarget = renderingData.cameraData.renderer.cameraColorTarget;
                cmd.SetGlobalTexture(RT_SSR_CameraTexture_ID, CameraColorTarget);

                using (new ProfilingScope(cmd, profilingSampler))
                {
                    cmd.SetViewProjectionMatrices(Matrix4x4.identity, Matrix4x4.identity);

                    GenerateHiZBuffers(cmd);

                    cmd.SetRenderTarget(new RenderTargetIdentifier[2] { new RenderTargetIdentifier(RT_SSR_Result1_ID), new RenderTargetIdentifier(RT_SSR_Result2_ID) }, new RenderTargetIdentifier(RT_SSR_Depth_None_ID));
                    cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial, 0, 0);

                    cmd.SetRenderTarget(RT_SSR_Resolved_ID);
                    cmd.SetGlobalTexture(RT_SSR_Result1_ID, RT_SSR_Result1_ID);
                    cmd.SetGlobalTexture(RT_SSR_Result2_ID, RT_SSR_Result2_ID);
                    cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial, 0, 1);
                    //时域降噪
                    if (Settings.TemporalFilter)
                    {
                        cmd.SetRenderTarget(RT_SSR_Denoised_ID);

                        cmd.SetGlobalTexture(RT_SSR_TemporalFilter_In_ID, RT_SSR_Resolved_ID);
                        SSRMaterial.SetTexture(RT_SSR_Denoised_Pre_ID, RT_SSR_Denoised_Pre);

                        cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial, 0, 2);
                        cmd.CopyTexture(RT_SSR_Denoised_ID, RT_SSR_Denoised_Pre);
                    }
                    else
                    {
                        cmd.Blit(RT_SSR_Resolved_ID, RT_SSR_Denoised_ID);
                    }
                    //拷贝到屏幕
                    cmd.SetGlobalTexture(GLOBAL_RT_Stochastic_SSR_ID, RT_SSR_Denoised_ID);

                    if (Settings.Debug)
                    {
                        cmd.Blit(RT_SSR_Denoised_ID, renderingData.cameraData.renderer.cameraColorTarget);
                    }
                    //cmd.Blit(RT_HiZMipmap_WithMipmap, renderingData.cameraData.renderer.cameraColorTarget);
                    //cmd.Blit(HiZMipmapBuffers_ID[5], renderingData.cameraData.renderer.cameraColorTarget);

                    //后处理结束
                    cmd.SetViewProjectionMatrices(camera.worldToCameraMatrix, camera.projectionMatrix);
                }

                context.ExecuteCommandBuffer(cmd);
                CommandBufferPool.Release(cmd);

            }
            public override void FrameCleanup(CommandBuffer cmd)
            {
                cmd.ReleaseTemporaryRT(RT_SSR_Result1_ID);
                cmd.ReleaseTemporaryRT(RT_SSR_Result2_ID);
                cmd.ReleaseTemporaryRT(RT_SSR_Depth_None_ID);
                cmd.ReleaseTemporaryRT(RT_SSR_Resolved_ID);
                cmd.ReleaseTemporaryRT(RT_SSR_Denoised_ID);
                DestroyHiZMipmap(cmd);
            }

        }
        private RenderPass RenderPass_Instance;
        public override void Create()
        {
            RenderPass_Instance = new RenderPass(Settings);
            RenderPass_Instance.renderPassEvent = Settings.passEvent + Settings.passEventOffset;
        }


        public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
        {
            renderer.EnqueuePass(RenderPass_Instance);
        }
    }
}


