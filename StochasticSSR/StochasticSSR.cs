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
public class StochasticSSR : ScriptableRendererFeature
{
    [System.Serializable]
    public class RenderBasicSetting
    {
        public string RenderPassName = "StochasticSSR";
        public RenderPassEvent passEvent = RenderPassEvent.BeforeRenderingPostProcessing;
        public int passEventOffset = 2;
        public bool FullPrecision = false;
        public bool Debug = false;

        public Texture test;
    }
    [System.Serializable]
    public class SSSRSettings
    {
        [InspectorToggleLeft] public bool HierarchicalZBuffer;
        [Range(1, 8)] public int HiZLevels = 4;
        [Range(0.0f, 1.0f)] public double MipmapBias_Width = 1.0f;
        [Range(0.0f, 1.0f)] public double MipmapBias_Height = 1.0f;
        [InspectorToggleLeft] public bool BinarySearch = true;
        [Range(1.0f, 3.0f)] public float BinarySearch_Tolerance = 1.4f;
        [InspectorToggleLeft] public bool BilinearDepthBuffer = true;
        public float Bilinear_Tolerance = 0.5f;
        public int BinarySearchIterations = 5;
        public float Thickness = 0.1f;
        [Range(0.005f, 1.0f)] public float StepLength = 0.1f;
        public int SamplesPerPixel = 2;
        [Range(0.0f,1.0f)] public float AngleBias=0.1f;
        [Range(0.0f, 1.0f)] public float MaxRoughness = 0.5f;
        [Range(0.0f, 1.0f)] public float MirrorReflectionThreshold = 0.05f;
        [InspectorToggleLeft] public bool TemporalFilter = true;
        [Range(0f, 1f)] public float TemporalFilterIntensity = 0.2f;
    }
    public RenderBasicSetting RenderBasic_Setting = new RenderBasicSetting();
    public SSSRSettings SSSR_Settings = new SSSRSettings();
    class RenderPass : ScriptableRenderPass
    {
        
        private static readonly int P_World2View_Matrix_ID = Shader.PropertyToID("World2View_Matrix");
        private static readonly int P_View2World_Matrix_ID = Shader.PropertyToID("View2World_Matrix");
        private static readonly int P_InvProjection_Matrix_ID = Shader.PropertyToID("InvProjection_Matrix");
        private static readonly int P_Projection_Matrix_ID = Shader.PropertyToID("Projection_Matrix");
        private static readonly int P_fov_ID = Shader.PropertyToID("fov");
        private static readonly int P_FarClipPlane_ID = Shader.PropertyToID("F");
        private static readonly int P_NearClipPlane_ID = Shader.PropertyToID("N");
        private static readonly int P_Bilinear_Tolerance_ID = Shader.PropertyToID("Bilinear_Tolerance");
        private static readonly int P_BinarySearch_Tolerance_ID = Shader.PropertyToID("BinarySearch_Tolerance");
        private static readonly int P_Thickness_ID = Shader.PropertyToID("THICKNESS");
        private static readonly int P_StepLength_ID = Shader.PropertyToID("STEPLENGTH");
        private static readonly int P_SamplesPerPixel_ID = Shader.PropertyToID("SAMPLESPERPIXEL");
        private static readonly int P_AngleBias_ID = Shader.PropertyToID("ANGLE_BIAS");
        private static readonly int P_MaxRoughness_ID = Shader.PropertyToID("MAXROUGHNESS");
        private static readonly int P_BinarySearchIterations_ID = Shader.PropertyToID("BINARYSEARCHITERATIONS");
        private static readonly int P_ScreenParams_ID = Shader.PropertyToID("RT_TexelSize");
        private static readonly int P_MirrorReflectionThreshold_ID = Shader.PropertyToID("MirrorReflectionThreshold");
        private static readonly int P_TemporalFilterIntensity_ID = Shader.PropertyToID("TemporalFilterIntensity");

        private static readonly int P_MaxMipLevels_ID = Shader.PropertyToID("MaxMipLevel");

        private static readonly int RT_SSR_CameraTexture_ID = Shader.PropertyToID("SSR_CameraTexture");
        private static readonly int RT_SSR_Result1_ID = Shader.PropertyToID("RT_SSR_Result1");
        private static readonly int RT_SSR_Result2_ID = Shader.PropertyToID("RT_SSR_Result2");
        private static readonly int RT_SSR_Depth_None_ID= Shader.PropertyToID("RT_None");
        private static readonly int RT_SSR_Resolved_ID = Shader.PropertyToID("RT_SSR_Resolve");
        private static readonly int RT_SSR_TemporalFilter_In_ID = Shader.PropertyToID("RT_SSR_TemporalFilter_In");
        private static readonly int RT_SSR_Denoised_ID = Shader.PropertyToID("RT_SSR_Denoised");
        private static readonly int RT_SSR_Denoised_Pre_ID = Shader.PropertyToID("RT_SSR_Denoised_Pre");
        private static readonly int GLOBAL_RT_Stochastic_SSR_ID = Shader.PropertyToID("Stochastic_SSR");

        private static readonly int[] RT_HiZMipmap_Levels_ID=new int[9] {
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

        private RenderBasicSetting RenderBasic_Setting;
        private SSSRSettings SSSR_Settings;
        private Material SSRMaterial;
        private RenderTexture RT_SSR_Denoised_Pre;

        private int[] HiZMipmapBuffers_ID;
        private int[] MipmapWidth;//0到max,max+1个
        private int[] MipmapHeight;
        private int MaxMipLevel = 0;
        private void InitializeHiZMipmap(int HiZLevels, Camera camera, double MipmapBias_Height, double MipmapBias_Width,CommandBuffer cmd)
        {
            int Mip0Width = (int)Math.Round(Math.Pow(2.0, Math.Floor(Math.Log(camera.pixelWidth, 2.0) + MipmapBias_Width)));
            int MaxMipMap_Width = (int)Math.Log(Mip0Width, 2.0);
            int Mip0Height = (int)Math.Round(Math.Pow(2.0, Math.Floor(Math.Log(camera.pixelHeight, 2.0) + MipmapBias_Height)));
            int MaxMipMap_Height = (int)Math.Log(Mip0Height, 2.0);
            MaxMipLevel = Math.Min(MaxMipMap_Width, MaxMipMap_Height);
            MaxMipLevel = Math.Min(MaxMipLevel, HiZLevels);

            MipmapWidth = new int[MaxMipLevel + 1];
            MipmapHeight = new int[MaxMipLevel + 1];
            HiZMipmapBuffers_ID = new int[MaxMipLevel + 1];

            MipmapHeight[0] = Mip0Height;
            MipmapWidth[0] = Mip0Width;
            HiZMipmapBuffers_ID[0] = Shader.PropertyToID("HiZMipmapBuffers_"+0);
            cmd.GetTemporaryRT(HiZMipmapBuffers_ID[0],MipmapWidth[0], MipmapHeight[0], 0, FilterMode.Point,RenderTextureFormat.RHalf);

            for (int i = 1; i <= MaxMipLevel; i++)
            {
                MipmapHeight[i] = MipmapHeight[i - 1] / 2;
                MipmapWidth[i] = MipmapWidth[i - 1] / 2;
                HiZMipmapBuffers_ID[i] = Shader.PropertyToID("HiZMipmapBuffers_" + i);
                cmd.GetTemporaryRT(HiZMipmapBuffers_ID[i], MipmapWidth[i], MipmapHeight[i], 0, FilterMode.Point, RenderTextureFormat.RHalf);

            }
        }
        private void GenerateHiZBuffers(CommandBuffer cmd)
        {
            var HiZMaterial = new Material(Shader.Find("PostProcess/HiZMipmap"));
            HiZMaterial.SetTexture("aaa",RenderBasic_Setting.test);

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
        }
        public void DestroyHiZMipmap(CommandBuffer cmd)
        {
            foreach (var A in HiZMipmapBuffers_ID)
            {
                cmd.ReleaseTemporaryRT(A);
            }
        }

        public RenderPass(RenderBasicSetting RenderBasic_Setting, SSSRSettings SSSR_Settings)
        {
            this.RenderBasic_Setting = RenderBasic_Setting;
            this.SSSR_Settings = SSSR_Settings;
        }
        public override void Configure(CommandBuffer cmd, RenderTextureDescriptor cameraTextureDescriptor)
        {
            //声明使用Normal
            ConfigureInput(ScriptableRenderPassInput.Depth | ScriptableRenderPassInput.Normal | ScriptableRenderPassInput.Motion);
        }
        public override void OnCameraSetup(CommandBuffer cmd, ref RenderingData renderingData)
        {
            var camera = renderingData.cameraData.camera;

            if (RT_SSR_Denoised_Pre is null || RT_SSR_Denoised_Pre.IsDestroyed())
            {
                RT_SSR_Denoised_Pre = new RenderTexture(camera.pixelWidth, camera.pixelHeight, 0, RenderTextureFormat.DefaultHDR);
                RT_SSR_Denoised_Pre.filterMode = FilterMode.Bilinear;
                RT_SSR_Denoised_Pre.Create();
            }
            if (camera.pixelWidth != RT_SSR_Denoised_Pre.width || camera.pixelHeight != RT_SSR_Denoised_Pre.height)
            {
                RT_SSR_Denoised_Pre.DiscardContents();
                RT_SSR_Denoised_Pre.Release();
                DestroyImmediate(RT_SSR_Denoised_Pre);
                RT_SSR_Denoised_Pre = null;

                RT_SSR_Denoised_Pre = new RenderTexture(camera.pixelWidth, camera.pixelHeight, 0, RenderTextureFormat.DefaultHDR);
                RT_SSR_Denoised_Pre.filterMode = FilterMode.Bilinear;
                RT_SSR_Denoised_Pre.Create();
            }
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get(RenderBasic_Setting.RenderPassName);
            Camera camera = renderingData.cameraData.camera;

            InitializeHiZMipmap(SSSR_Settings.HiZLevels,camera,SSSR_Settings.MipmapBias_Height,SSSR_Settings.MipmapBias_Width,cmd);

            SSRMaterial = new Material(Shader.Find("PostProcess/StochasticSSR"));
            Vector4 ScreenParams = new Vector4(1/ camera.pixelWidth,1/ camera.pixelHeight, camera.pixelWidth,camera.pixelHeight);
            SSRMaterial.SetVector(P_ScreenParams_ID, ScreenParams);

            SSRMaterial.SetMatrix(P_World2View_Matrix_ID, camera.worldToCameraMatrix);
            SSRMaterial.SetMatrix(P_View2World_Matrix_ID, camera.cameraToWorldMatrix);
            SSRMaterial.SetMatrix(P_Projection_Matrix_ID, camera.projectionMatrix);
            SSRMaterial.SetMatrix(P_InvProjection_Matrix_ID, camera.projectionMatrix.inverse);
            SSRMaterial.SetFloat(P_fov_ID, camera.fieldOfView);
            SSRMaterial.SetFloat(P_NearClipPlane_ID, camera.nearClipPlane);
            SSRMaterial.SetFloat(P_FarClipPlane_ID, camera.farClipPlane);

            cmd.SetGlobalTexture("aaa", RenderBasic_Setting.test);

            if (SSSR_Settings.BinarySearch) { SSRMaterial.EnableKeyword("BINARY_SEARCH"); }
            if (SSSR_Settings.BilinearDepthBuffer) { SSRMaterial.EnableKeyword("BILINEAR_DEPTHBUFFER"); }
            if (RenderBasic_Setting.FullPrecision) { SSRMaterial.EnableKeyword("FULL_PRECISION_SSR"); }

            SSRMaterial.SetFloat(P_Bilinear_Tolerance_ID, SSSR_Settings.Bilinear_Tolerance);
            SSRMaterial.SetFloat(P_BinarySearch_Tolerance_ID,SSSR_Settings.BinarySearch_Tolerance);
            SSRMaterial.SetFloat(P_Thickness_ID,SSSR_Settings.Thickness);
            SSRMaterial.SetFloat(P_StepLength_ID, SSSR_Settings.StepLength*0.1f);
            SSRMaterial.SetInt(P_SamplesPerPixel_ID, SSSR_Settings.SamplesPerPixel);
            SSRMaterial.SetFloat(P_AngleBias_ID, SSSR_Settings.AngleBias);
            SSRMaterial.SetFloat(P_MaxRoughness_ID,SSSR_Settings.MaxRoughness);
            SSRMaterial.SetInt(P_BinarySearchIterations_ID, SSSR_Settings.BinarySearchIterations);
            SSRMaterial.SetFloat(P_MirrorReflectionThreshold_ID, SSSR_Settings.MirrorReflectionThreshold);
            SSRMaterial.SetFloat(P_TemporalFilterIntensity_ID, SSSR_Settings.TemporalFilterIntensity);

            cmd.GetTemporaryRT(RT_SSR_Result1_ID,camera.pixelWidth, camera.pixelHeight, 0,FilterMode.Point, RenderTextureFormat.ARGBHalf);
            cmd.GetTemporaryRT(RT_SSR_Result2_ID, camera.pixelWidth, camera.pixelHeight, 0, FilterMode.Point, RenderTextureFormat.ARGBHalf);
            cmd.GetTemporaryRT(RT_SSR_Depth_None_ID, camera.pixelWidth, camera.pixelHeight, 8, FilterMode.Point, RenderTextureFormat.Depth);
            cmd.GetTemporaryRT(RT_SSR_Resolved_ID, camera.pixelWidth, camera.pixelHeight, 0, FilterMode.Point, RenderTextureFormat.DefaultHDR);
            cmd.GetTemporaryRT(RT_SSR_Denoised_ID, camera.pixelWidth, camera.pixelHeight, 0, FilterMode.Point, RenderTextureFormat.DefaultHDR);

            var CameraColorTarget = renderingData.cameraData.renderer.cameraColorTarget;
            cmd.SetGlobalTexture(RT_SSR_CameraTexture_ID, CameraColorTarget);

            using (new ProfilingScope(cmd, profilingSampler))
            {
                cmd.SetViewProjectionMatrices(Matrix4x4.identity, Matrix4x4.identity);

                GenerateHiZBuffers(cmd);
                for(int i = 0; i <= HiZMipmapBuffers_ID.Length-1; i++)
                {
                    cmd.SetGlobalTexture(RT_HiZMipmap_Levels_ID[i], HiZMipmapBuffers_ID[i]);
                }
                SSRMaterial.SetInt(P_MaxMipLevels_ID,MaxMipLevel);

                cmd.SetRenderTarget(new RenderTargetIdentifier[2]{ new RenderTargetIdentifier(RT_SSR_Result1_ID), new RenderTargetIdentifier(RT_SSR_Result2_ID) },new RenderTargetIdentifier(RT_SSR_Depth_None_ID));
                cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial,0,0);

                cmd.SetRenderTarget(RT_SSR_Resolved_ID);
                cmd.SetGlobalTexture(RT_SSR_Result1_ID, RT_SSR_Result1_ID);
                cmd.SetGlobalTexture(RT_SSR_Result2_ID, RT_SSR_Result2_ID);
                cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial, 0, 1);
                //时域降噪
                if (SSSR_Settings.TemporalFilter)
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

                if (RenderBasic_Setting.Debug)
                {
                    cmd.Blit(RT_SSR_Denoised_ID, renderingData.cameraData.renderer.cameraColorTarget);
                }
                cmd.Blit(HiZMipmapBuffers_ID[0],renderingData.cameraData.renderer.cameraColorTarget);

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
        RenderPass_Instance = new RenderPass(RenderBasic_Setting, SSSR_Settings);
        RenderPass_Instance.renderPassEvent = RenderBasic_Setting.passEvent + RenderBasic_Setting.passEventOffset;
    }


    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(RenderPass_Instance);
    }
}


