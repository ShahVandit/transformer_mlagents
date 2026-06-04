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
    private float batteryDrainRate = 1f / 2500f;
    private int ptReachCount = 0;
    private int rechargeCount = 0;

    // === Episode tracking ===
    private int episodeSteps = 0;
    private bool episodeEnded = false;
    private int firstPtReachStep = -1;

    // === Visuals ===
    private Renderer renderGround;
    private Renderer renderTarget;

    // === Logging ===
    private LearningDirector LearningBoard;

    // === Path visualization ===
    private LineRenderer line;
    private int points;

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

        // Path visualization
        Transform lineRenderObj = transform.parent.Find("LineRender");
        if (lineRenderObj != null)
        {
            line = lineRenderObj.GetComponent<LineRenderer>();
            if (line != null)
            {
                line.startColor = new Color(0.5f, 0f, 1f);
                line.endColor = new Color(0.5f, 0f, 1f);
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
            Debug.Log($"[SimpleTask1] Training mode: {trainingMode} (from {configPath})");
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
        episodeSteps = 0;
        episodeEnded = false;
        firstPtReachStep = -1;

        // Reset path visualization
        if (line != null) line.positionCount = 0;
        points = 1;

        MaxStep = 50000;

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

    // ==================== OBSERVATIONS (7 dims) ====================

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

            Debug.Log($"PT obs=({relPT.x/400f:F3}, {relPT.z/400f:F3})");
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // === 3. BATTERY LEVEL (1 dim) ===
        sensor.AddObservation(battery);

        // === 4. BATTERY STATION DIRECTION (2 dims) ===
        if (tfBatteryStation != null)
        {
            Vector3 relBS = tfBatteryStation.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relBS.x / 400f);
            sensor.AddObservation(relBS.z / 400f);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // === TOTAL: 2 + 2 + 1 + 2 = 7 vector dims ===
    }

    // ==================== ACTIONS ====================

    public override void OnActionReceived(ActionBuffers actions)
    {
        // === Battery drain (Task2 and Combined only; Task1 battery is constant) ===
        if (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined)
        {
            battery -= batteryDrainRate;
            battery = Mathf.Max(battery, 0f);

            if (battery <= 0f)
            {
                EndEpisodeCustom("battery_death");
                return;
            }
        }

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

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag.Equals("Target"))
        {
            TargetReached();
        }
        else if (collision.gameObject.tag.Equals("BatteryStation"))
        {
            BatteryStationReached();
        }
        else if (collision.gameObject.tag.Equals("Geometry"))
        {
            EndEpisodeCustom("obstacle");
        }
    }

    // ==================== TARGET (PURSUIT) ====================

    public void TargetReached()
    {
        if (ptReachCount == 0) firstPtReachStep = episodeSteps;
        ptReachCount++;

        // Task 1 reward (pursuit modes)
        if (trainingMode == TrainingMode.Task1 || trainingMode == TrainingMode.Combined)
        {
            AddReward(2f);
        }

        // Visual feedback
        // renderGround.material.color = Color.green;

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
            float rx = Random.Range(-200f, 200f);
            float rz = Random.Range(-200f, 200f);
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

    private const float rechargeThreshold = 0.2f;

    private void BatteryStationReached()
    {
        // Only reward genuine recharges: battery must have drained below threshold.
        // Prevents oscillation exploit (step out, step back in, farm +0.5).
        // With drain=1/2500 and threshold=0.2, the agent must spend ~2000 steps
        // away from base before a recharge qualifies — enforcing ~20 cycles/episode.
        if (battery < rechargeThreshold)
        {
            battery = 1.0f;
            rechargeCount++;

            if (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined)
            {
                AddReward(1f);
            }
        }
        else if (battery < 1.0f)
        {
            // Still recharge (keep alive), but no reward — not a meaningful drain
            battery = 1.0f;
        }
    }

    // ==================== EPISODE END ====================

    private void EndEpisodeCustom(string reason)
    {
        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        float ptRate = episodeSteps > 0 ? (float)ptReachCount / episodeSteps * 1000f : 0f;
        string episodeLog = $"[{timestamp}] [EPISODE END] reason={reason} | mode={trainingMode} " +
            $"| totalSteps={episodeSteps} | reward={GetCumulativeReward():F1} " +
            $"| ptReaches={ptReachCount} | recharges={rechargeCount} " +
            $"| batteryAtEnd={battery:F3} | ptRate={ptRate:F2} | firstPtStep={firstPtReachStep}";

        if (LearningBoard != null)
            LearningBoard.UpdateStats(ptReachCount, rechargeCount);

        switch (reason)
        {
            case "obstacle":
                if (LearningBoard != null) LearningBoard.IncreaseFailed();
                AddReward(-5f);
                renderGround.material.color = Color.red;
                break;

            case "timeout":
                if (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined)
                    AddReward(5f);
                if (LearningBoard != null) LearningBoard.IncreaseSuccess();
                break;

            case "battery_death":
                AddReward(-5f);
                if (LearningBoard != null) LearningBoard.IncreaseFailed();
                renderGround.material.color = Color.yellow;
                break;

            default:
                Debug.LogWarning($"EndEpisodeCustom called with unknown reason: {reason}");
                break;
        }

        // Build stats AFTER Increase* so eps reflects the updated global count
        if (LearningBoard != null)
        {
            string statsLog = $"[{timestamp}] [EPISODE STATS] ptVisits={ptReachCount} | recharges={rechargeCount} | steps={episodeSteps} | mode={trainingMode} | avgPtVisits={LearningBoard.AvgPtVisits:F2} | avgRecharges={LearningBoard.AvgRecharges:F2} | eps={LearningBoard.TotalEpisodes}";
            LearningBoard.WriteEpisodeLog(episodeLog);
            LearningBoard.WriteEpisodeLog(statsLog);
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
        if (line == null || points < 1) return;
        Vector3 pt = transform.position;

        if (points == 1)
        {
            line.positionCount = points;
            line.SetPosition(points - 1, pt);
            points++;
        }
        else if (line.positionCount >= points - 1 && Vector3.Distance(line.GetPosition(points - 2), pt) >= 0.5f)
        {
            if (pt != Vector3.zero)
            {
                line.positionCount = points;
                line.SetPosition(points - 1, pt);
                points++;
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
                float randomX = Random.Range(-200f, 200f);
                float randomZ = Random.Range(-200f, 200f);
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
            float rx = Random.Range(-200f, 200f);
            float rz = Random.Range(-200f, 200f);
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
                float randomX = Random.Range(-200f, 200f);
                float randomZ = Random.Range(-200f, 200f);
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
