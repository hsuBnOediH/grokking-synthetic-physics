using System.Collections;
using System.IO;
using System.Diagnostics;
using UnityEngine;
using Debug = UnityEngine.Debug;

/// <summary>
/// V3 DataGenerator — reads episode_design.csv (produced by design_episodes.py)
/// and samples each episode's physics parameters within the pre-assigned sub-range.
/// This guarantees a perfectly balanced IID / Near-OOD / Far-OOD split by design.
/// </summary>
public class DataGenerator : MonoBehaviour
{
    [Header("Files")]
    public string episodeDesignFile = "episode_design.csv";
    public string saveDirectory     = "GeneratedDataV3";

    [Header("Episode Config")]
    public int framesPerEpisode = 100;

    [Header("Scene References")]
    public Camera        agentCamera;
    public Transform     pendulumSystem;
    public Rigidbody     pendulumRb;
    public MeshRenderer  bobRenderer;
    public Transform     pendulumRod;

    // Hardcoded rendering / physics constants
    private const int   Resolution          = 64;
    private const int   PhysicsStepsPerFrame = 3;
    private const float CamRotationRange    = 5f;
    private const float CamElevationRange   = 2f;
    private const float MinElevation        = 10f;
    private const float MaxElevation        = 80f;

    // Gravity / damping color encoding bounds (must match design_episodes.py PARAM_RANGES)
    private const float MinGravity  = 4.0f;
    private const float MaxGravity  = 14.0f;
    private const float MinDamping  = 0.01f;
    private const float MaxDamping  = 0.5f;

    // Per-episode episode design (loaded from CSV)
    private struct EpisodeConfig
    {
        public float LengthLo, LengthHi;
        public float AngleLo,  AngleHi;
        public float GravityLo, GravityHi;
        public float DampingLo, DampingHi;
        public float AngVelLo, AngVelHi;
    }
    private EpisodeConfig[] episodeConfigs;

    // Cached geometry
    private Vector3 originalRodScale;
    private Vector3 originalBobLocalPos;

    // Per-episode values (written to CSV)
    private float epGravity, epLength, epInitAngVel, epDamping;

    // I/O
    private RenderTexture renderTexture;
    private Texture2D     texture2D;
    private StreamWriter  csvWriter;
    private Stopwatch     stopwatch;

    // -----------------------------------------------------------------------

    void Start()
    {
        LoadEpisodeDesign();

        if (pendulumRod != null)
            originalRodScale = pendulumRod.localScale;
        originalBobLocalPos = pendulumRb.transform.localPosition;

        InitializeIO();
        InitializeRendering();
        stopwatch = Stopwatch.StartNew();
        StartCoroutine(DataGenerationLoop());
    }

    private void LoadEpisodeDesign()
    {
        if (!File.Exists(episodeDesignFile))
        {
            Debug.LogError($"Episode design file not found: {episodeDesignFile}");
            return;
        }

        string[] lines = File.ReadAllLines(episodeDesignFile);
        // lines[0] = header, lines[1..] = data rows (already sorted by episode_id)
        episodeConfigs = new EpisodeConfig[lines.Length - 1];

        for (int i = 1; i < lines.Length; i++)
        {
            string[] p = lines[i].Split(',');
            // CSV columns: episode_id(0), combo_id(1), n_ood_dims(2), split(3),
            //              length_lo(4), length_hi(5), angle_lo(6), angle_hi(7),
            //              gravity_lo(8), gravity_hi(9), damping_lo(10), damping_hi(11),
            //              angvel_lo(12), angvel_hi(13)
            episodeConfigs[i - 1] = new EpisodeConfig
            {
                LengthLo  = float.Parse(p[4]),  LengthHi  = float.Parse(p[5]),
                AngleLo   = float.Parse(p[6]),  AngleHi   = float.Parse(p[7]),
                GravityLo = float.Parse(p[8]),  GravityHi = float.Parse(p[9]),
                DampingLo = float.Parse(p[10]), DampingHi = float.Parse(p[11]),
                AngVelLo  = float.Parse(p[12]), AngVelHi  = float.Parse(p[13]),
            };
        }

        Debug.Log($"Loaded {episodeConfigs.Length} episode configs from {episodeDesignFile}");
    }

    private void InitializeIO()
    {
        if (!Directory.Exists(saveDirectory)) Directory.CreateDirectory(saveDirectory);
        string csvPath = Path.Combine(saveDirectory, "ground_truth.csv");
        csvWriter = new StreamWriter(csvPath, false);
        csvWriter.WriteLine("Episode,Frame,Damping,Gravity,Length,InitAngularVelocity,Angle,AngularVelocity,Camera_X,Camera_Y,Camera_Z");
    }

    private void InitializeRendering()
    {
        renderTexture = new RenderTexture(Resolution, Resolution, 24);
        agentCamera.targetTexture = renderTexture;
        texture2D = new Texture2D(Resolution, Resolution, TextureFormat.RGB24, false);
    }

    private IEnumerator DataGenerationLoop()
    {
        int numEpisodes = episodeConfigs.Length;
        int totalFrames = numEpisodes * framesPerEpisode;
        int framesDone  = 0;

        Debug.Log($"=== V3 Data Generation: {numEpisodes} episodes x {framesPerEpisode} frames = {totalFrames} total ===");
        Debug.Log($"=== Save directory: {Path.GetFullPath(saveDirectory)} ===");

        for (int ep = 0; ep < numEpisodes; ep++)
        {
            ResetEnvironment(ep);
            yield return new WaitForFixedUpdate();

            for (int frame = 0; frame < framesPerEpisode; frame++)
            {
                yield return new WaitForEndOfFrame();

                float currentAngle = pendulumSystem.eulerAngles.x;
                float angVel = pendulumRb.angularVelocity.x;
                Vector3 camPos = agentCamera.transform.position;
                csvWriter.WriteLine($"{ep},{frame},{epDamping},{epGravity},{epLength},{epInitAngVel},{currentAngle},{angVel},{camPos.x},{camPos.y},{camPos.z}");

                SaveCameraView(Path.Combine(saveDirectory, $"ep{ep}_frame{frame}.png"));

                float deltaAzimuth = Random.Range(-CamRotationRange, CamRotationRange);
                agentCamera.transform.RotateAround(pendulumSystem.position, Vector3.up, deltaAzimuth);

                float deltaElevation = Random.Range(-CamElevationRange, CamElevationRange);
                agentCamera.transform.RotateAround(pendulumSystem.position, agentCamera.transform.right, deltaElevation);

                ClampCameraElevation();
                agentCamera.transform.LookAt(pendulumSystem.position);

                for (int step = 0; step < PhysicsStepsPerFrame; step++)
                    yield return new WaitForFixedUpdate();

                framesDone++;
            }

            float pct     = (float)(ep + 1) / numEpisodes * 100f;
            float elapsed = (float)stopwatch.Elapsed.TotalSeconds;
            float fps     = framesDone / elapsed;
            float eta     = (totalFrames - framesDone) / fps;
            Debug.Log($"[{pct:F1}%] Episode {ep + 1}/{numEpisodes} | {fps:F0} fps | ETA: {eta:F0}s");
        }

        float totalTime = (float)stopwatch.Elapsed.TotalSeconds;
        Debug.Log($"=== DONE! {totalFrames} frames in {totalTime:F1}s ({totalFrames / totalTime:F0} fps) ===");
        Debug.Log($"=== Saved to: {Path.GetFullPath(saveDirectory)} ===");

        Physics.gravity = new Vector3(0, -9.81f, 0);
        csvWriter.Close();

#if UNITY_EDITOR
        UnityEditor.EditorApplication.isPlaying = false;
#endif
    }

    private void ResetEnvironment(int ep)
    {
        EpisodeConfig cfg = episodeConfigs[ep];

        // D3: Gravity — encoded as bob Hue
        epGravity = Random.Range(cfg.GravityLo, cfg.GravityHi);
        Physics.gravity = new Vector3(0, -epGravity, 0);

        // D4: Damping — encoded as bob Saturation
        epDamping = Random.Range(cfg.DampingLo, cfg.DampingHi);
        pendulumRb.angularDamping = epDamping;

        // HSV color encoding
        float hue = Mathf.InverseLerp(MinGravity, MaxGravity, epGravity);
        float sat = Mathf.Lerp(0.3f, 1.0f, Mathf.InverseLerp(MinDamping, MaxDamping, epDamping));
        MaterialPropertyBlock props = new MaterialPropertyBlock();
        props.SetColor("_BaseColor", Color.HSVToRGB(hue, sat, 1.0f));
        bobRenderer.SetPropertyBlock(props);

        // D1: Length — geometrically observable
        epLength = Random.Range(cfg.LengthLo, cfg.LengthHi);
        if (pendulumRod != null)
            pendulumRod.localScale = new Vector3(originalRodScale.x, epLength, originalRodScale.z);
        float lengthRatio = epLength / originalRodScale.y;
        pendulumRb.transform.localPosition = new Vector3(0, originalBobLocalPos.y * lengthRatio, 0);

        // D2: Initial angle — geometrically observable
        float angle = Random.Range(cfg.AngleLo, cfg.AngleHi);
        pendulumSystem.rotation = Quaternion.Euler(angle, 0, 0);

        // D5: Initial angular velocity — purely latent
        epInitAngVel = Random.Range(cfg.AngVelLo, cfg.AngVelHi);
        pendulumRb.linearVelocity  = Vector3.zero;
        pendulumRb.angularVelocity = new Vector3(epInitAngVel, 0, 0);
    }

    private void ClampCameraElevation()
    {
        Vector3 dir = agentCamera.transform.position - pendulumSystem.position;
        float elevation = Mathf.Asin(dir.y / dir.magnitude) * Mathf.Rad2Deg;
        if (elevation < MinElevation || elevation > MaxElevation)
        {
            float correction = Mathf.Clamp(elevation, MinElevation, MaxElevation) - elevation;
            agentCamera.transform.RotateAround(pendulumSystem.position, agentCamera.transform.right, correction);
        }
    }

    private void SaveCameraView(string path)
    {
        RenderTexture.active = renderTexture;
        texture2D.ReadPixels(new Rect(0, 0, Resolution, Resolution), 0, 0);
        texture2D.Apply();
        RenderTexture.active = null;
        File.WriteAllBytes(path, texture2D.EncodeToPNG());
    }

    void OnDestroy()
    {
        if (csvWriter != null) csvWriter.Close();
        Physics.gravity = new Vector3(0, -9.81f, 0);
    }
}
