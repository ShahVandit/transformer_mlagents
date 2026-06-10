using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class SimpleTask1 : Agent
{
    // === Training mode ===
    private enum TrainingMode { Task1, Task2, Combined }
    private TrainingMode trainingMode = TrainingMode.Combined;

    // === Agent references ===
    private Transform tfAgent;
    private Rigidbody rbAgent;
    private Transform tfTarget;
    private Transform tfBatteryStation;
    public float maxSpeed = 20f;

    // === Battery system ===
    private float battery = 1.0f;
    private float batteryDrainRate = 1f / 4000f;
    private int ptReachCount = 0;
    private int rechargeCount = 0;
    private bool isInBatteryStation = false;
    private bool isCharging = false;
    private float chargeProgress = 0f;
    private float rechargeRatePerStep = 0.004f;
    private float batteryAtChargeStart = 0f;
    private bool validChargeCycle = false;

    // === Episode tracking ===
    private int episodeSteps = 0;
    private bool episodeEnded = false;
    private int firstPtReachStep = -1;
    private List<int> ptReachSteps = new List<int>();

    // === Visuals ===
    private Renderer renderGround;
    private Renderer renderTarget;

    // === Logging ===
    private LearningDirector LearningBoard;

    // === Targets===
    public int maxTargetsToReach = 5;

    // === Path visualization (one LineRenderer per segment) ===
    private LineRenderer[] segmentLines = new LineRenderer[3];
    private int activeSegment = 0;
    private int[] segmentPoints = new int[] { 1, 1, 1 };
    private static readonly Color[] segmentColors = new Color[]
    {
        new Color(0.1f, 0.9f, 0.1f),  // segment 0 → PT1: green
        new Color(1f, 0.75f, 0f),      // segment 1 → PT2: yellow
        new Color(1f, 0.25f, 0f)       // segment 2 → PT3: orange-red
    };

    // === Wind (kept for MDP consistency) ===
    public float rotation;
    public float windfactor;
    public Vector3 windForce;

    public override void Initialize()
    {
        tfAgent = GetComponent<Transform>();
        rbAgent = GetComponent<Rigidbody>();
        tfTarget = transform.parent.Find("Target").GetComponent<Transform>();

        tfBatteryStation = transform.parent.Find("BatteryStation");
        if (tfBatteryStation == null)
            Debug.LogError("[SimpleTask1] BatteryStation not found in scene hierarchy.");

        renderGround = transform.parent.Find("Ground").GetComponent<Renderer>();
        renderTarget = tfTarget.GetComponent<Renderer>();

        // Path visualization — find 3 segment LineRenderers
        string[] lineNames = { "LineRender", "LineRender2", "LineRender3" };
        for (int i = 0; i < 3; i++)
        {
            Transform obj = transform.parent.Find(lineNames[i]);
            if (obj != null)
            {
                segmentLines[i] = obj.GetComponent<LineRenderer>();
                if (segmentLines[i] != null)
                {
                    segmentLines[i].startColor = segmentColors[i];
                    segmentLines[i].endColor = segmentColors[i];
                    segmentLines[i].material.color = segmentColors[i];
                    Gradient g = new Gradient();
                    g.SetKeys(
                        new GradientColorKey[] {
                            new GradientColorKey(segmentColors[i], 0f),
                            new GradientColorKey(segmentColors[i], 1f)
                        },
                        new GradientAlphaKey[] {
                            new GradientAlphaKey(1f, 0f),
                            new GradientAlphaKey(1f, 1f)
                        }
                    );
                    segmentLines[i].colorGradient = g;
                }
            }
        }

        ReadTrainingMode();
    }

    void Start()
    {
        LearningBoard = GameObject.Find("LearningDirector")?.GetComponent<LearningDirector>();
        if (LearningBoard == null)
            Debug.LogError("LearningDirector not found or missing LearningDirector component.");
    }

    private void ReadTrainingMode()
    {
        // Try config relative to project root: <projectRoot>/config/current_run_config.json
        string configPath = System.IO.Path.GetFullPath(
            System.IO.Path.Combine(Application.dataPath, "..", "config", "current_run_config.json"));

        if (!System.IO.File.Exists(configPath))
        {
            // Fallback: try one level up (for builds)
            configPath = System.IO.Path.GetFullPath(
                System.IO.Path.Combine(Application.dataPath, "..", "..", "config", "current_run_config.json"));
        }

        if (System.IO.File.Exists(configPath))
        {
            string json = System.IO.File.ReadAllText(configPath);
            var match = System.Text.RegularExpressions.Regex.Match(
                json, "\"training_mode\"\\s*:\\s*\"([^\"]+)\"");
            if (match.Success)
            {
                string mode = match.Groups[1].Value.ToLower();
                switch (mode)
                {
                    case "task1": trainingMode = TrainingMode.Task1; break;
                    case "task2": trainingMode = TrainingMode.Task2; break;
                    case "combined": trainingMode = TrainingMode.Combined; break;
                    default:
                        Debug.LogWarning($"[SimpleTask1] Unknown training_mode '{mode}', defaulting to Combined");
                        trainingMode = TrainingMode.Combined;
                        break;
                }
            }
            var maxStepsMatch = System.Text.RegularExpressions.Regex.Match(
                json, "\"max_steps\"\\s*:\\s*(\\d+)");
            if (maxStepsMatch.Success)
                MaxStep = int.Parse(maxStepsMatch.Groups[1].Value);

            var drainMatch = System.Text.RegularExpressions.Regex.Match(
                json, "\"battery_drain_rate\"\\s*:\\s*\"([^\"]+)\"");
            if (drainMatch.Success)
            {
                var parts = drainMatch.Groups[1].Value.Split('/');
                if (parts.Length == 2)
                    batteryDrainRate = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture)
                        / float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                else
                    batteryDrainRate = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
            }

            var rechargeMatch = System.Text.RegularExpressions.Regex.Match(
                json, "\"recharge_rate\"\\s*:\\s*\"([^\"]+)\"");
            if (rechargeMatch.Success)
            {
                var parts = rechargeMatch.Groups[1].Value.Split('/');
                if (parts.Length == 2)
                    rechargeRatePerStep = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture)
                        / float.Parse(parts[1], System.Globalization.CultureInfo.InvariantCulture);
                else
                    rechargeRatePerStep = float.Parse(parts[0], System.Globalization.CultureInfo.InvariantCulture);
            }

            var thresholdMatch = System.Text.RegularExpressions.Regex.Match(
                json, "\"recharge_threshold\"\\s*:\\s*([\\d.]+)");
            if (thresholdMatch.Success)
                rechargeThreshold = float.Parse(thresholdMatch.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture);

            int batteryLifeSteps = batteryDrainRate > 0 ? Mathf.RoundToInt(1f / batteryDrainRate) : 0;
            int rechargeSteps = rechargeRatePerStep > 0 ? Mathf.RoundToInt(1f / rechargeRatePerStep) : 0;
            Debug.Log($"[SimpleTask1] Training mode: {trainingMode}, MaxStep: {MaxStep}, batteryDrainRate: {batteryDrainRate:F6} (~{batteryLifeSteps} steps/charge), rechargeRate: {rechargeRatePerStep:F6} (~{rechargeSteps} steps to full), rechargeThreshold: {rechargeThreshold:F2} (from {configPath})");
        }
        else
        {
            Debug.LogWarning("[SimpleTask1] Config not found, defaulting to Combined mode");
        }
    }

    public override void OnEpisodeBegin()
    {
        // Reset physics
        rbAgent.linearVelocity = Vector3.zero;
        rbAgent.angularVelocity = Vector3.zero;
        tfAgent.eulerAngles = Vector3.zero;

        // Reset battery & counters
        battery = 1.0f;
        ptReachCount = 0;
        rechargeCount = 0;
        isInBatteryStation = false;
        isCharging = false;
        chargeProgress = 0f;
        batteryAtChargeStart = 0f;
        validChargeCycle = false;
        episodeSteps = 0;
        episodeEnded = false;
        firstPtReachStep = -1;
        ptReachSteps.Clear();

        // Reset path visualization
        for (int i = 0; i < 3; i++)
            if (segmentLines[i] != null) segmentLines[i].positionCount = 0;
        activeSegment = 0;
        segmentPoints = new int[] { 1, 1, 1 };

        // Randomize environment
        RandomizeObstaclePositionsAndSizes();
        RandomizeTargetPosition();
        RandomizeBatteryStationPosition();
        RandomizeAgentPosition();

        // Ensure objects are active
        tfTarget.gameObject.SetActive(true);
        if (tfBatteryStation != null)
            tfBatteryStation.gameObject.SetActive(true);
        this.gameObject.SetActive(true);

        StartCoroutine(RevertMaterial());
    }

    // ==================== OBSERVATIONS (10 dims) ====================

    public override void CollectObservations(VectorSensor sensor)
    {
        // === 1. VELOCITY (2 dims) ===
        sensor.AddObservation(rbAgent.linearVelocity.x / maxSpeed);
        sensor.AddObservation(rbAgent.linearVelocity.z / maxSpeed);
        // === 2. PT DIRECTION (2 dims) ===
        if (tfTarget != null && tfTarget.gameObject.activeSelf)
        {
            Vector3 relPT = tfTarget.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relPT.x / 400f);
            sensor.AddObservation(relPT.z / 400f);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // === 3. BATTERY LEVEL (1 dim) ===
        sensor.AddObservation(trainingMode == TrainingMode.Task1 ? 1.0f : battery);

        // === 4. BATTERY STATION DIRECTION (2 dims) ===
        if (trainingMode != TrainingMode.Task1 && tfBatteryStation != null)
        {
            Vector3 relBS = tfBatteryStation.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relBS.x / 400f);
            sensor.AddObservation(relBS.z / 400f);
            // Debug.Log($"X: {relBS.x / 400f}, Z:{relBS.z / 400f}");
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // === 5. SECOND BATTERY STATION (3 dims, unused — padding for BS2 dir x,z + BS2 distance) ===
        sensor.AddObservation(0f); // BS2 dir x
        sensor.AddObservation(0f); // BS2 dir z
        sensor.AddObservation(0f); // BS2 distance
        // === 6. DISTANCES (2 dims) ===
        if (tfTarget != null && tfTarget.gameObject.activeSelf)
        {
            Vector3 relPT = tfTarget.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relPT.magnitude / 400f);
        }
        else
        {
            sensor.AddObservation(0f);
        }

        if (trainingMode != TrainingMode.Task1 && tfBatteryStation != null)
        {
            Vector3 relBS = tfBatteryStation.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relBS.magnitude / 400f);
        }
        else
        {
            sensor.AddObservation(0f);
        }

        // === 7. CHARGING STATE (1 dim) ===
        // needscharging is reserved for delayed recharge logic; keep it shared now.
        sensor.AddObservation(
            trainingMode == TrainingMode.Task1 ? 0f : (battery < rechargeThreshold ? 1f : 0f)
        );

        // === 8. CURRENT COUNT (1 dim) ===
        sensor.AddObservation(ptReachCount);
        // === 9. NORMALIZED STEP COUNT (1 dim) ===
        sensor.AddObservation((float)StepCount / MaxStep);

        // === TOTAL: 2 + 2 + 1 + 2 + 3 + 2 + 1 + 1 + 1 = 15 vector dims ===
    }

    // ==================== ACTIONS ====================

    public override void OnActionReceived(ActionBuffers actions)
    {
        // === Battery drain (Task2 and Combined only; Task1 battery is constant) ===
        // {
        battery -= batteryDrainRate;
        battery = Mathf.Max(battery, 0f);
        if (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined)
        {
            if (battery <= 0f)
            {
                EndEpisodeCustom("battery_dead");
                return;
            }
        }
        // Debug.Log($"Battery:{battery}, Reward:{GetCumulativeReward()}");
        // === Step penalty (all modes) ===
        AddReward(-0.001f);

        // === Movement ===
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        Vector3 moveDirection = new Vector3(moveX, 0, moveZ);
        if (moveDirection.magnitude > 0.01f)
            moveDirection = moveDirection.normalized;

        rbAgent.linearVelocity = moveDirection * maxSpeed;

        // === Timeout check ===
        if (this.StepCount >= MaxStep)
        {
            EndEpisodeCustom("timeout");
        }
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var ContinuousActionsOut = actionsOut.ContinuousActions;

        if (Input.GetKey(KeyCode.D))
            ContinuousActionsOut[0] = 1;
        else if (Input.GetKey(KeyCode.A))
            ContinuousActionsOut[0] = -1;
        else
            ContinuousActionsOut[0] = 0;

        if (Input.GetKey(KeyCode.W))
            ContinuousActionsOut[1] = 1;
        else if (Input.GetKey(KeyCode.S))
            ContinuousActionsOut[1] = -1;
        else
            ContinuousActionsOut[1] = 0;
    }

    // ==================== COLLISIONS ====================

    private void OnTriggerEnter(Collider collision)
    {
        if (collision.gameObject.tag.Equals("Target"))
        {
            TargetReached();
        }
    }

    private void OnCollisionEnter(Collision collision)
    {
        if (episodeEnded) return;
        if (collision.gameObject.tag.Equals("BatteryStation"))
        {
            isInBatteryStation = true;
        }
        else if (collision.gameObject.tag.Equals("Geometry"))
        {
            EndEpisodeCustom("obstacle");
        }
    }

    private void OnCollisionStay(Collision collision)
    {
        if (episodeEnded) return;
        if (collision.gameObject.tag.Equals("BatteryStation") &&
            (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined))
        {
            isInBatteryStation = true;
            BatteryStationReached();
            // Penalize sitting in the station while not actively charging
            if (!isCharging)
                AddReward(-0.005f);
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.tag.Equals("BatteryStation"))
        {
            // Pay proportional recharge reward for a valid cycle, only on exit
            if (validChargeCycle &&
                (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined))
            {
                float chargeReward = Mathf.Clamp(battery - batteryAtChargeStart, 0f, 1f);
                AddReward(chargeReward);
            }
            validChargeCycle = false;
            isInBatteryStation = false;
            isCharging = false;
        }
    }

    // ==================== TARGET (PURSUIT) ====================

    public void TargetReached()
    {
        if (ptReachCount == 0) firstPtReachStep = episodeSteps;
        ptReachSteps.Add(episodeSteps);
        ptReachCount++;
        if (activeSegment < 2) activeSegment++;

        // Task 1 reward (pursuit modes)
        if (trainingMode == TrainingMode.Task1 || trainingMode == TrainingMode.Combined)
        {
            AddReward(10f);
        }

        if ((trainingMode == TrainingMode.Task1 || trainingMode == TrainingMode.Combined) &&  ptReachCount >= maxTargetsToReach)
        {
            EndEpisodeCustom("success");
            return;
        }
        // Respawn PT at new random position (always, all modes)
        RespawnTarget();
    }

    private void RespawnTarget()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        bool isValid = false;
        Vector3 newPos = Vector3.zero;
        int attempt = 0;

        while (!isValid && attempt < 100)
        {
            float rx = Random.Range(-185f, 185f);
            float rz = Random.Range(-185f, 185f);
            float fy = tfTarget.localPosition.y;
            newPos = new Vector3(rx, fy, rz);

            isValid = true;
            if (obstaclesParent != null)
            {
                foreach (Transform obstacle in obstaclesParent)
                {
                    if (Vector3.Distance(newPos, obstacle.localPosition) < 50f)
                    { isValid = false; break; }
                }
            }
            if (isValid && tfBatteryStation != null &&
                Vector3.Distance(newPos, tfBatteryStation.localPosition) < 30f)
                isValid = false;
            if (isValid && Vector3.Distance(newPos, tfAgent.localPosition) < 30f)
                isValid = false;

            attempt++;
        }

        if (isValid)
            tfTarget.localPosition = newPos;
        else
            Debug.LogWarning("[SimpleTask1] Failed to find valid PT respawn position.");

        tfTarget.gameObject.SetActive(true);
    }

    // ==================== BATTERY STATION (ENERGY) ====================

    private float rechargeThreshold = 0.5f;

    private void BatteryStationReached()
    {
        // Start a valid charge cycle only when below threshold
        if (!isCharging && battery < rechargeThreshold)
        {
            isCharging = true;
            batteryAtChargeStart = battery;
            validChargeCycle = true;
        }

        // Once charging, keep going until full (no reward while charging — paid on exit)
        if (isCharging)
        {
            battery = Mathf.Min(1.0f, battery + rechargeRatePerStep);
            chargeProgress = Mathf.InverseLerp(rechargeThreshold, 1.0f, battery);

            if (battery >= 1.0f)
            {
                rechargeCount++;
                chargeProgress = 1.0f;
                isCharging = false;
            }
        }
    }

    // ==================== EPISODE END ====================

    private void EndEpisodeCustom(string reason)
    {
        if (episodeEnded) return;
        episodeEnded = true;

        if (LearningBoard != null)
            LearningBoard.UpdateStats(ptReachCount, rechargeCount);

        switch (reason)
        {
            case "success":
                AddReward(5f);
                if (LearningBoard != null)
                    LearningBoard.IncreaseSuccess();
                break;
            case "obstacle":
                if (LearningBoard != null) LearningBoard.IncreaseFailed();
                AddReward(-5f);
                renderGround.material.color = Color.red;
                break;

            case "timeout":
                AddReward(-10f);
                if (trainingMode == TrainingMode.Combined)
                    AddReward((maxTargetsToReach - ptReachCount) * -10f);
                if (LearningBoard != null) LearningBoard.IncreaseTimeOut();
                break;

            case "battery_dead":
                AddReward(-5f);
                if (LearningBoard != null) LearningBoard.IncreaseBatteryDeath();
                renderGround.material.color = Color.yellow;
                if (LearningBoard != null)
                {
                    string ts = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
                    LearningBoard.WriteEpisodeLog($"[{ts}] [BATTERY DEAD] steps={episodeSteps} | ptReaches={ptReachCount} | recharges={rechargeCount} | batteryAtDeath={battery:F3} | reward={GetCumulativeReward():F1}");
                }
                break;

            default:
                Debug.LogWarning($"EndEpisodeCustom called with unknown reason: {reason}");
                break;
        }

        // Build logs AFTER terminal reward and Increase* so both are captured
        if (LearningBoard != null)
        {
            string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
            float ptRate = episodeSteps > 0 ? (float)ptReachCount / episodeSteps * 1000f : 0f;
            int batteryLifeSteps = batteryDrainRate > 0 ? Mathf.RoundToInt(1f / batteryDrainRate) : 0;
            // Debug.Log($"[Battery] life={batteryLifeSteps}steps | atEnd={battery:F3} | recharges={rechargeCount} | reason={reason}");

            string episodeEndLog = $"[{timestamp}] [EPISODE END] reason={reason} | mode={trainingMode} | totalSteps={episodeSteps} | reward={GetCumulativeReward():F1} | ptReaches={ptReachCount} | recharges={rechargeCount} | batteryAtEnd={battery:F3} | batteryLife={batteryLifeSteps}steps | ptRate={ptRate:F2} | firstPtStep={firstPtReachStep}";
            LearningBoard.WriteEpisodeLog(episodeEndLog);

            string episodeStatsLog = $"[{timestamp}] [EPISODE STATS] ptVisits={ptReachCount} | recharges={rechargeCount} | steps={episodeSteps} | mode={trainingMode} | avgPtVisits={LearningBoard.AvgPtVisits:F2} | avgRecharges={LearningBoard.AvgRecharges:F2} | eps={LearningBoard.TotalEpisodes}";
            LearningBoard.WriteEpisodeLog(episodeStatsLog);

            // Per-target step log: steps to reach each PT and delta between consecutive ones
            var ptDeltas = new System.Text.StringBuilder();
            int prev = 0;
            for (int i = 0; i < ptReachSteps.Count; i++)
            {
                int delta = ptReachSteps[i] - prev;
                ptDeltas.Append($"PT{i + 1}@{ptReachSteps[i]}(+{delta})");
                if (i < ptReachSteps.Count - 1) ptDeltas.Append(" ");
                prev = ptReachSteps[i];
            }
            if (ptReachSteps.Count > 0)
            {
                string ptStepsLog = $"[{timestamp}] [PT STEPS] {ptDeltas}";
                LearningBoard.WriteEpisodeLog(ptStepsLog);
            }
        }

        EndEpisode();
    }

    // ==================== FIXED UPDATE ====================

    void FixedUpdate()
    {
        episodeSteps++;
        AddPoints();
    }

    private void AddPoints()
    {
        if (activeSegment >= segmentLines.Length) return;
        LineRenderer line = segmentLines[activeSegment];
        if (line == null) return;

        int pts = segmentPoints[activeSegment];
        Vector3 pt = transform.position;

        if (pts == 1)
        {
            line.positionCount = pts;
            line.SetPosition(pts - 1, pt);
            segmentPoints[activeSegment]++;
        }
        else if (line.positionCount >= pts - 1 && Vector3.Distance(line.GetPosition(pts - 2), pt) >= 0.5f)
        {
            if (pt != Vector3.zero)
            {
                line.positionCount = pts;
                line.SetPosition(pts - 1, pt);
                segmentPoints[activeSegment]++;
            }
        }
    }

    // ==================== VISUAL HELPERS ====================

    IEnumerator RevertMaterial()
    {
        yield return new WaitForSeconds(0.01f);
        renderGround.material.color = Color.white;
        renderTarget.material.color = Color.blue;
    }

    // ==================== RANDOMIZATION ====================

    private void RandomizeObstaclePositionsAndSizes()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        if (obstaclesParent != null)
        {
            int gridSize = 4;
            float cellSize = 100f; // fixed cell size regardless of grid count
            float gridOrigin = -(gridSize * cellSize) / 2f; // centers 400x400 grid in 500x500 arena

            int obstacleIndex = 0;
            foreach (Transform obstacle in obstaclesParent)
            {
                int row = obstacleIndex / gridSize;
                int col = obstacleIndex % gridSize;

                float baseX = gridOrigin + col * cellSize + cellSize / 2;
                float baseZ = gridOrigin + row * cellSize + cellSize / 2;

                float randomOffsetX = Random.Range(-cellSize / 4, cellSize / 4);
                float randomOffsetZ = Random.Range(-cellSize / 4, cellSize / 4);
                float fixedY = obstacle.localPosition.y;

                obstacle.localPosition = new Vector3(baseX + randomOffsetX, fixedY, baseZ + randomOffsetZ);

                float randomScaleX = Random.Range(30f, 50f);
                float randomScaleZ = Random.Range(30f, 50f);
                obstacle.localScale = new Vector3(randomScaleX, 100, randomScaleZ);

                obstacleIndex++;
                if (obstacleIndex >= gridSize * gridSize)
                    break;
            }
        }
        else
        {
            Debug.LogError("Obstacles parent object not found.");
        }
    }

    private void RandomizeTargetPosition()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        Transform targetTransform = transform.parent.Find("Target");

        if (obstaclesParent != null && targetTransform != null)
        {
            bool isPositionValid = false;
            Vector3 newTargetPosition = Vector3.zero;
            int attempt = 0;

            while (!isPositionValid && attempt < 100)
            {
                float randomX = Random.Range(-185f, 185f);
                float randomZ = Random.Range(-185f, 185f);
                float fixedY = targetTransform.localPosition.y;
                newTargetPosition = new Vector3(randomX, fixedY, randomZ);

                isPositionValid = true;
                foreach (Transform obstacle in obstaclesParent)
                {
                    if (Vector3.Distance(newTargetPosition, obstacle.localPosition) < 50f)
                    {
                        isPositionValid = false;
                        break;
                    }
                }
                attempt++;
            }

            if (isPositionValid)
            {
                targetTransform.localPosition = newTargetPosition;
                tfTarget = targetTransform;
                tfTarget.gameObject.SetActive(true);
            }
            else
            {
                Debug.LogError("Failed to find a valid position for the target.");
            }
        }
        else
        {
            Debug.LogError("Obstacles or Target object not found.");
        }
    }

    private void RandomizeBatteryStationPosition()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        if (obstaclesParent == null || tfBatteryStation == null) return;

        bool isValid = false;
        Vector3 newPos = Vector3.zero;
        int attempt = 0;

        while (!isValid && attempt < 100)
        {
            float rx = Random.Range(-185f, 185f);
            float rz = Random.Range(-185f, 185f);
            float fy = tfBatteryStation.localPosition.y;
            newPos = new Vector3(rx, fy, rz);

            isValid = true;
            foreach (Transform obstacle in obstaclesParent)
            {
                if (Vector3.Distance(newPos, obstacle.localPosition) < 50f)
                { isValid = false; break; }
            }
            // Don't stack on PT
            if (isValid && Vector3.Distance(newPos, tfTarget.localPosition) < 50f)
                isValid = false;
            // Don't stack on agent
            if (isValid && Vector3.Distance(newPos, tfAgent.localPosition) < 50f)
                isValid = false;

            attempt++;
        }

        if (isValid)
            tfBatteryStation.localPosition = newPos;
        else
            Debug.LogError("Failed to find valid battery station position.");
    }

    private void RandomizeAgentPosition()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");

        if (obstaclesParent != null)
        {
            bool isPositionValid = false;
            Vector3 newAgentPosition = Vector3.zero;
            int attempt = 0;

            while (!isPositionValid && attempt < 100)
            {
                float randomX = Random.Range(-185f, 185f);
                float randomZ = Random.Range(-185f, 185f);
                float fixedY = 20;
                newAgentPosition = new Vector3(randomX, fixedY, randomZ);

                isPositionValid = true;
                foreach (Transform obstacle in obstaclesParent)
                {
                    if (Vector3.Distance(newAgentPosition, obstacle.localPosition) < 50f)
                    {
                        isPositionValid = false;
                        break;
                    }
                }
                attempt++;
            }

            if (isPositionValid)
                transform.localPosition = newAgentPosition;
            else
                Debug.LogError("Failed to find a valid position for the agent.");
        }
        else
        {
            Debug.LogError("Obstacles parent not found.");
        }
    }
}
