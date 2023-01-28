using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Experimental.Rendering;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;
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
    }
    [System.Serializable]
    public class RayMarchSetting
    {
        [InspectorToggleLeft] public bool BinarySearch = true;
        [Range(1.0f, 3.0f)] public float BinarySearch_Tolerance = 1.4f;
        [InspectorToggleLeft] public bool BilinearDepthBuffer = true;
        public float Bilinear_Tolerance = 0.5f;
        public float Thickness = 0.1f;
        [Range(0.005f, 1.0f)] public float StepLength = 0.1f;
        public int SamplesPerPixel = 2;
    }
    public RenderBasicSetting Render_Basic_Setting = new RenderBasicSetting();
    public RayMarchSetting RayMarch_Setting = new RayMarchSetting();
    class RenderPass : ScriptableRenderPass
    {
        private static readonly int SSR_Dest_ID = Shader.PropertyToID("_SSR_Dest");

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

        private static readonly int RT_SSR_CameraTexture_ID = Shader.PropertyToID("SSR_CameraTexture");

        public RenderBasicSetting Render_Basic_Setting;
        public RayMarchSetting RayMarch_Setting;
        public Material SSRMaterial=new Material(Shader.Find("PostProcess/StochasticSSR"));

        public RenderPass(RenderBasicSetting Render_Basic_Setting, RayMarchSetting RayMarch_Setting)
        {
            this.Render_Basic_Setting = Render_Basic_Setting;
            this.RayMarch_Setting = RayMarch_Setting;
        }
        public override void Execute(ScriptableRenderContext context, ref RenderingData renderingData)
        {
            CommandBuffer cmd = CommandBufferPool.Get(Render_Basic_Setting.RenderPassName);
            Camera camera = renderingData.cameraData.camera;

            SSRMaterial.SetMatrix(P_World2View_Matrix_ID, camera.worldToCameraMatrix);
            SSRMaterial.SetMatrix(P_View2World_Matrix_ID, camera.cameraToWorldMatrix);
            SSRMaterial.SetMatrix(P_Projection_Matrix_ID, camera.projectionMatrix);
            SSRMaterial.SetMatrix(P_InvProjection_Matrix_ID, camera.projectionMatrix.inverse);
            SSRMaterial.SetFloat(P_fov_ID, camera.fieldOfView);
            SSRMaterial.SetFloat(P_NearClipPlane_ID, camera.nearClipPlane);
            SSRMaterial.SetFloat(P_FarClipPlane_ID, camera.farClipPlane);

            if (RayMarch_Setting.BinarySearch) { SSRMaterial.EnableKeyword("BINARY_SEARCH"); }
            if (RayMarch_Setting.BilinearDepthBuffer) { SSRMaterial.EnableKeyword("BILINEAR_DEPTHBUFFER"); }
            if (Render_Basic_Setting.FullPrecision) { SSRMaterial.EnableKeyword("FULL_PRECISION_SSR"); }
            SSRMaterial.SetFloat(P_Bilinear_Tolerance_ID, RayMarch_Setting.Bilinear_Tolerance);
            SSRMaterial.SetFloat(P_BinarySearch_Tolerance_ID,RayMarch_Setting.BinarySearch_Tolerance);
            SSRMaterial.SetFloat(P_Thickness_ID,RayMarch_Setting.Thickness);
            SSRMaterial.SetFloat(P_StepLength_ID, RayMarch_Setting.StepLength*0.1f);
            SSRMaterial.SetInt(P_SamplesPerPixel_ID, RayMarch_Setting.SamplesPerPixel);

            cmd.GetTemporaryRT(SSR_Dest_ID, camera.pixelWidth, camera.pixelHeight, 0, FilterMode.Point, GraphicsFormat.B10G11R11_UFloatPack32);
            var CameraColorTarget = renderingData.cameraData.renderer.cameraColorTarget;
            cmd.SetGlobalTexture(RT_SSR_CameraTexture_ID, CameraColorTarget);

            using (new ProfilingScope(cmd, profilingSampler))
            {
                cmd.SetViewProjectionMatrices(Matrix4x4.identity, Matrix4x4.identity);

                cmd.SetRenderTarget(SSR_Dest_ID);
                cmd.DrawMesh(RenderingUtils.fullscreenMesh, Matrix4x4.identity, SSRMaterial,0);
                cmd.Blit(SSR_Dest_ID, CameraColorTarget);

                cmd.SetViewProjectionMatrices(camera.worldToCameraMatrix, camera.projectionMatrix);
            }
            context.ExecuteCommandBuffer(cmd);
            CommandBufferPool.Release(cmd);
        }
        public override void FrameCleanup(CommandBuffer cmd)
        {
            cmd.ReleaseTemporaryRT(SSR_Dest_ID);
        }
    }
    private RenderPass RenderPass_Instance;
    public override void Create()
    {
        RenderPass_Instance = new RenderPass(Render_Basic_Setting, RayMarch_Setting);
        RenderPass_Instance.renderPassEvent = Render_Basic_Setting.passEvent + Render_Basic_Setting.passEventOffset;
    }


    public override void AddRenderPasses(ScriptableRenderer renderer, ref RenderingData renderingData)
    {
        renderer.EnqueuePass(RenderPass_Instance);
    }
}


