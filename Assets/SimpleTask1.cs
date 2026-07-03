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
    // Split recharge payout: a small reward at the 0.9 checkpoint (dense
    // learning signal so the critic sees positive return-to-go as soon as
    // charging works) plus a larger reward at full. The large full chunk makes
    // finishing strictly best — leaving at 0.9 forfeits rechargeFullReward,
    // far more than the ~0.025 nav-step cost of charging 0.9->1.0. Total still
    // sums to 2.0 to keep the per-recharge reward scale consistent with soups.
    private bool rechargePartialPaid = false;
    private bool rechargeFullPaid = false;
    private float rechargeBonusLevel = 0.9f;
    private float rechargePartialReward = 0.5f;
    private float rechargeFullReward = 1.5f;
    // V3 ablation switch. "urgency" (default) = V2 behavior: IsRechargeUrgent()
    // gates a split bonus and drives obs dim 96. "none" = no threshold anywhere:
    // dim 96 is held at 0 and charging earns a proportional, ungated reward
    // (kCharge * battery added per step), so the policy must learn WHEN to
    // charge from raw battery + distances instead of being handed an urgency flag.
    private string chargeRewardMode = "urgency";
    private float kCharge = 2.5f;
    private bool rechargeLocked = false;
    private bool wasChargeCycleThisVisit = false;
    // Freezes charging_state obs to 1 during an active valid (urgent) cycle so
    // the policy is not told "you're done" mid-charge when IsRechargeUrgent()
    // relaxes as battery rises. Reset on station exit.
    private bool chargingStateFrozen = false;
    private int topUpCount = 0;
    private float initialBatteryMin = 0.4f;
    private float initialBatteryMax = 1.0f;
    private float obstaclePenalty = 50f;
    private float bsSpawnRange = 50f;
    // Reachability-aware charging: charge only when the battery can't safely
    // cover "reach PT, then reach BS" (distance-aware), instead of a flat
    // threshold. navStepsPerUnit is the calibrated steps/unit from
    // calibrate_nav_ratio.py (k_p90=3.91, conservative; ideal straight line
    // is 2.5). chargeHardFloor is an absolute safety net so battery_dead
    // can't balloon if the distance estimate is optimistic.
    private float navStepsPerUnit = 3.91f;
    private float chargeHardFloor  = 0.00f;
    // Exact straight-line travel per battery-drain step; computed in
    // Initialize from maxSpeed, fixed timestep, and decision settings.
    private float unitsPerStep = 0.4f;
    private int chargeCycleStarts = 0;
    private int validChargeExits = 0;
    private int fullChargeCycles = 0;
    private int chargingSteps = 0;
    private int stationIdleSteps = 0;
    // "Leave once charged" shaping: time spent in the BS collider while NOT
    // actively charging (i.e. full/locked) is loitering. The penalty escalates
    // with consecutive idle steps this visit so the agent departs promptly
    // instead of camping on the station. Shapes exit behavior only — not charge
    // timing. Reset on station exit and episode start.
    private int idleStepsThisVisit = 0;
    private float idlePenalty = 0.005f;     // per-step base, config: idle_penalty
    private int idleEscalCap = 20;          // penalty caps at idlePenalty * cap
    private float batteryAtStart = 1f;
    private float totalChargeGained = 0f;
    private float minChargeStartBattery = 1f;
    private float maxChargeExitBattery = 0f;

    // === Episode tracking ===
    private int episodeSteps = 0;
    private bool episodeEnded = false;
    private int firstPtReachStep = -1;
    private List<int> ptReachSteps = new List<int>();
    // Straight-line distance of each leg (agent->PT at the moment that PT
    // spawned), pushed when the PT is reached. Paired with ptReachSteps to
    // calibrate steps-per-unit-distance from battery-off (task1) inference.
    private List<float> ptLegDists = new List<float>();
    private float currentLegDist = 0f;
    private List<string> chargeEventSummaries = new List<string>();
    private List<string> episodeEventTimeline = new List<string>();

    // === Visuals ===
    private Renderer renderGround;
    private Renderer renderTarget;
    private TMPro.TMP_Text batteryLabel;

    // === Logging ===
    private LearningDirector LearningBoard;

    // === Targets===
    public int maxTargetsToReach = 5;

    // === Path visualization (one LineRenderer per segment) ===
    private LineRenderer[] segmentLines;
    private int activeSegment = 0;
    private int[] segmentPoints;
    private Color[] segmentColors;

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

        // Search all descendants of the env parent (BatteryLabel may be nested under Ground, etc.)
        foreach (var t in transform.parent.GetComponentsInChildren<TMPro.TMP_Text>(true))
        {
            if (t.name == "BatteryLabel")
            {
                batteryLabel = t;
                break;
            }
        }
        if (batteryLabel == null)
            Debug.LogWarning("[SimpleTask1] 'BatteryLabel' TMP_Text not found anywhere under env parent.");

        ReadTrainingMode();

        // Path visualization — one LineRenderer per target, named LineRender / LineRender2 / LineRender3 ...
        segmentLines  = new LineRenderer[maxTargetsToReach];
        segmentPoints = new int[maxTargetsToReach];
        segmentColors = new Color[maxTargetsToReach];
        for (int i = 0; i < maxTargetsToReach; i++) segmentPoints[i] = 1;
        for (int i = 0; i < maxTargetsToReach; i++)
        {
            string lineName = i == 0 ? "LineRender" : $"LineRender{i + 1}";
            Transform obj = transform.parent.Find(lineName);
            if (obj != null)
            {
                segmentLines[i] = obj.GetComponent<LineRenderer>();
                // Neutralize the material's own color tint (called once → one
                // material instance per renderer, no per-frame leak).
                if (segmentLines[i] != null)
                    segmentLines[i].material.color = Color.white;
            }
        }

        // Distance covered per OnActionReceived call (= per battery-drain
        // step). Velocity is set directly to maxSpeed, so this is exact.
        var decisionRequester = GetComponent<Unity.MLAgents.DecisionRequester>();
        int stepsPerAction = (decisionRequester != null && !decisionRequester.TakeActionsBetweenDecisions)
            ? decisionRequester.DecisionPeriod
            : 1;
        unitsPerStep = maxSpeed * Time.fixedDeltaTime * stepsPerAction;
    }

    void Start()
    {
        LearningBoard = GameObject.Find("LearningDirector")?.GetComponent<LearningDirector>();
        if (LearningBoard == null)
            Debug.LogError("LearningDirector not found or missing LearningDirector component.");
    }

    private void UpdateBatteryLabel()
    {
        if (batteryLabel == null) return;
        batteryLabel.text  = chargeRewardMode == "none"
            ? $"bat={battery:F2} step={StepCount} chg={isCharging}"
            : $"bat={battery:F2} step={StepCount} need={chargingStateFrozen || IsRechargeUrgent()}";
        batteryLabel.color = battery < 0.3f ? Color.red
                           : battery < 0.6f ? Color.yellow
                           : Color.white;
    }

    // Reachability-aware charge trigger. Returns true when the battery cannot
    // safely cover "reach the current PT, then reach the BS" at the calibrated
    // nav rate, OR when it drops below an absolute hard floor. Replaces the old
    // flat "battery < rechargeThreshold". Distance-aware: a nearby PT keeps the
    // agent committed to navigation; a far PT triggers charging earlier.
    private bool IsRechargeUrgent()
    {
        if (trainingMode == TrainingMode.Task1) return false;          // nav-only: never charges
        if (battery < chargeHardFloor) return true;                    // absolute safety net
        if (tfBatteryStation == null || tfTarget == null) return false;

        // battery burned per straight-line unit = drain/step * steps/unit
        float batPerUnit = batteryDrainRate * navStepsPerUnit;
        float distToPT   = Vector3.Distance(tfAgent.localPosition, tfTarget.localPosition);
        float distPTtoBS = Vector3.Distance(tfTarget.localPosition, tfBatteryStation.localPosition);
        float needed     = (distToPT + distPTtoBS) * batPerUnit;       // reach PT, then reach BS

        return battery < needed;
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

            // Tunable env/reward params: missing keys keep the code default,
            // so old configs stay valid.
            float ReadFloat(string key, float fallback)
            {
                var m = System.Text.RegularExpressions.Regex.Match(
                    json, $"\"{key}\"\\s*:\\s*(-?[\\d.]+)");
                return m.Success
                    ? float.Parse(m.Groups[1].Value, System.Globalization.CultureInfo.InvariantCulture)
                    : fallback;
            }
            rechargeBonusLevel = ReadFloat("recharge_bonus_level", rechargeBonusLevel);
            rechargePartialReward = ReadFloat("recharge_partial_reward", rechargePartialReward);
            rechargeFullReward = ReadFloat("recharge_full_reward", rechargeFullReward);
            initialBatteryMin = ReadFloat("initial_battery_min", initialBatteryMin);
            initialBatteryMax = ReadFloat("initial_battery_max", initialBatteryMax);
            obstaclePenalty = ReadFloat("obstacle_penalty", obstaclePenalty);
            bsSpawnRange = ReadFloat("bs_spawn_range", bsSpawnRange);
            maxTargetsToReach = (int)ReadFloat("max_targets", maxTargetsToReach);
            navStepsPerUnit = ReadFloat("nav_steps_per_unit", navStepsPerUnit);
            chargeHardFloor = ReadFloat("charge_hard_floor", chargeHardFloor);
            kCharge = ReadFloat("k_charge", kCharge);
            idlePenalty = ReadFloat("idle_penalty", idlePenalty);
            var chargeModeMatch = System.Text.RegularExpressions.Regex.Match(
                json, "\"charge_reward_mode\"\\s*:\\s*\"([^\"]+)\"");
            if (chargeModeMatch.Success)
                chargeRewardMode = chargeModeMatch.Groups[1].Value.ToLower();
            Debug.Log($"[SimpleTask1] charge_reward_mode={chargeRewardMode}, k_charge={kCharge:F2}");

            int batteryLifeSteps = batteryDrainRate > 0 ? Mathf.RoundToInt(1f / batteryDrainRate) : 0;
            int rechargeSteps = rechargeRatePerStep > 0 ? Mathf.RoundToInt(1f / rechargeRatePerStep) : 0;
            Debug.Log($"[SimpleTask1] Training mode: {trainingMode}, MaxStep: {MaxStep}, batteryDrainRate: {batteryDrainRate:F6} (~{batteryLifeSteps} steps/charge), rechargeRate: {rechargeRatePerStep:F6} (~{rechargeSteps} steps to full), rechargeThreshold: {rechargeThreshold:F2}, bonus: +{rechargePartialReward:F1}@{rechargeBonusLevel:F2}/+{rechargeFullReward:F1}@full, initBattery: [{initialBatteryMin:F2},{initialBatteryMax:F2}], obstaclePenalty: -{obstaclePenalty:F0}, bsRange: ±{bsSpawnRange:F0}, maxTargets: {maxTargetsToReach} (from {configPath})");
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

        // Battery is finalized after spawn positions are known (see below),
        // so the random floor can guarantee the BS is reachable.
        battery = 1.0f;
        ptReachCount = 0;
        rechargeCount = 0;
        isInBatteryStation = false;
        isCharging = false;
        chargeProgress = 0f;
        batteryAtChargeStart = 0f;
        validChargeCycle = false;
        rechargePartialPaid = false;
        rechargeFullPaid = false;
        rechargeLocked = false;
        idleStepsThisVisit = 0;
        wasChargeCycleThisVisit = false;
        chargingStateFrozen = false;
        topUpCount = 0;
        chargeCycleStarts = 0;
        validChargeExits = 0;
        fullChargeCycles = 0;
        chargingSteps = 0;
        stationIdleSteps = 0;
        totalChargeGained = 0f;
        minChargeStartBattery = 1f;
        maxChargeExitBattery = 0f;
        episodeSteps = 0;
        episodeEnded = false;
        firstPtReachStep = -1;
        ptReachSteps.Clear();
        ptLegDists.Clear();
        currentLegDist = 0f;
        chargeEventSummaries.Clear();
        episodeEventTimeline.Clear();

        // Reset path visualization and assign fresh dark distinct colors for this episode.
        // Hues are evenly spaced around the wheel then shifted by a random offset so each
        // episode looks different while keeping adjacent legs visually distinct.
        float hueOffset = Random.value;
        for (int i = 0; i < maxTargetsToReach; i++)
        {
            float hue = (hueOffset + (float)i / maxTargetsToReach) % 1f;
            segmentColors[i] = Color.HSVToRGB(hue, 0.95f, 0.75f);
        }
        for (int i = 0; i < maxTargetsToReach; i++)
        {
            segmentPoints[i] = 1;
            if (segmentLines[i] == null) continue;
            segmentLines[i].positionCount = 0;
            segmentLines[i].startColor = segmentColors[i];
            segmentLines[i].endColor   = segmentColors[i];
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
        activeSegment = 0;

        // Randomize environment
        RandomizeObstaclePositionsAndSizes();
        RandomizeTargetPosition();
        RandomizeBatteryStationPosition();
        RandomizeAgentPosition();

        // Initial leg (agent spawn -> PT1): measured now that both the agent
        // and the first target are placed. Subsequent legs are set in RespawnTarget.
        currentLegDist = Vector3.Distance(tfAgent.localPosition, tfTarget.localPosition);

        // Randomize starting battery (battery modes), including below the
        // recharge threshold so the charge-or-pursue decision is encountered
        // early and often. Floor = 1.2x the battery needed to reach the BS
        // from spawn, so no episode is unwinnable at start.
        if (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined)
        {
            float distToBS = Vector3.Distance(tfAgent.localPosition, tfBatteryStation.localPosition);
            float stepsToBS = distToBS / unitsPerStep;
            float reachableFloor = Mathf.Clamp(
                1.2f * stepsToBS * batteryDrainRate, initialBatteryMin, initialBatteryMax);
            battery = Random.Range(reachableFloor, initialBatteryMax);
            batteryAtStart = battery;
        }

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
        // sensor.AddObservation(trainingMode == TrainingMode.Task1 ? 1.0f : battery);
        sensor.AddObservation(battery);

        // === 4. BATTERY STATION DIRECTION (2 dims) ===
        // BS obs are now live in all modes (incl. Task1) so a fine-tune sees a
        // real, varying battery-station signal. Only guard is null-safety.
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

        if (tfBatteryStation != null)
        {
            Vector3 relBS = tfBatteryStation.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relBS.magnitude / 400f);
        }
        else
        {
            sensor.AddObservation(0f);
        }

        // === 7. CHARGING STATE (1 dim) ===
        // Reachability-aware trigger (distance to PT-then-BS) instead of a flat
        // battery threshold — same single dim, smarter value. See IsRechargeUrgent.
        // Frozen to 1 during an active valid cycle so the policy stays committed
        // to filling up even as IsRechargeUrgent() relaxes mid-charge.
        // V3 ablation ("none"): dim 96 carries no urgency/threshold signal — the
        // policy must infer charge timing from battery (dim 88) + PT/BS distances.
        // Kept as a constant-0 dim (not removed) so obs stays 99-dim and the only
        // change vs V2 is this dimension's information content.
        if (chargeRewardMode == "none")
            sensor.AddObservation(0f);
        else
            sensor.AddObservation((chargingStateFrozen || IsRechargeUrgent()) ? 1f : 0f);

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
        // === Step reward: survival signal for Task2, speed incentive for Task1/Combined ===
        AddReward(trainingMode == TrainingMode.Task2 ? 0.0005f : -0.001f);



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
            // "Charge then leave": once charging stops (full/locked) further time
            // in the station is loitering. Penalty escalates with consecutive idle
            // steps this visit so the agent learns to depart promptly instead of
            // camping on the BS — the incorrect behavior seen in earlier Task2
            // runs. Escalation is capped to stay bounded.
            if (!isCharging)
            {
                stationIdleSteps++;
                idleStepsThisVisit++;
                AddReward(-idlePenalty * Mathf.Min(idleStepsThisVisit, idleEscalCap));
            }
        }
    }

    private void OnCollisionExit(Collision collision)
    {
        if (collision.gameObject.tag.Equals("BatteryStation"))
        {
            // No partial exit reward: the only charge payout is the near-full
            // bonus (paid in BatteryStationReached). Leaving a valid cycle
            // early therefore earns nothing, which discourages "charge a
            // little and leave" without an explicit penalty. Exits are only
            // logged here for analysis.
            if (validChargeCycle &&
                (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined))
            {
                float chargeGain = Mathf.Clamp(battery - batteryAtChargeStart, 0f, 1f);
                validChargeExits++;
                totalChargeGained += chargeGain;
                maxChargeExitBattery = Mathf.Max(maxChargeExitBattery, battery);
                if (battery >= 0.999f)
                    fullChargeCycles++;
                chargeEventSummaries.Add($"{batteryAtChargeStart:F3}->{battery:F3}(+{chargeGain:F3})");
                episodeEventTimeline.Add($"CH{validChargeExits}@{batteryAtChargeStart:F3}->{battery:F3}(+{chargeGain:F3})");
            }
            else if (battery > batteryAtChargeStart + 0.005f && wasChargeCycleThisVisit &&
                     (trainingMode == TrainingMode.Task2 || trainingMode == TrainingMode.Combined))
            {
                // Unrewarded opportunistic top-up — log it so emergence is visible
                topUpCount++;
                episodeEventTimeline.Add($"TU@{batteryAtChargeStart:F3}->{battery:F3}");
            }
            wasChargeCycleThisVisit = false;
            validChargeCycle = false;
            chargingStateFrozen = false;
            isInBatteryStation = false;
            isCharging = false;
            rechargeLocked = false;
            idleStepsThisVisit = 0;
        }
    }

    // ==================== TARGET (PURSUIT) ====================

    public void TargetReached()
    {
        if (ptReachCount == 0) firstPtReachStep = episodeSteps;
        ptReachSteps.Add(episodeSteps);
        ptLegDists.Add(currentLegDist);   // distance of the leg just completed
        ptReachCount++;
        episodeEventTimeline.Add($"PT{ptReachCount}@{episodeSteps}(b={battery:F3})");
        if (activeSegment < maxTargetsToReach - 1)
        {
            activeSegment++;
            // Activate the new segment's color now that it starts drawing.
            if (segmentLines[activeSegment] != null)
            {
                segmentLines[activeSegment].startColor = segmentColors[activeSegment];
                segmentLines[activeSegment].endColor   = segmentColors[activeSegment];
                segmentLines[activeSegment].material.color = segmentColors[activeSegment];
                Gradient g = new Gradient();
                g.SetKeys(
                    new GradientColorKey[] {
                        new GradientColorKey(segmentColors[activeSegment], 0f),
                        new GradientColorKey(segmentColors[activeSegment], 1f)
                    },
                    new GradientAlphaKey[] {
                        new GradientAlphaKey(1f, 0f),
                        new GradientAlphaKey(1f, 1f)
                    }
                );
                segmentLines[activeSegment].colorGradient = g;
            }
        }

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

        // Next leg starts here: agent is at the PT it just reached.
        currentLegDist = Vector3.Distance(tfAgent.localPosition, tfTarget.localPosition);

        tfTarget.gameObject.SetActive(true);
    }

    // ==================== BATTERY STATION (ENERGY) ====================

    private float rechargeThreshold = 0.5f;

    private void BatteryStationReached()
    {
        // Charging works at any battery level (opportunistic top-ups stay
        // possible — a smart agent passing the BS tops up to save a future
        // trip). Reward eligibility (validChargeCycle) is gated on
        // IsRechargeUrgent(): only a genuinely-needed charge earns the bonus.
        // Camping is prevented not by blocking starts but by locking on full
        // (below): once topped out, no restart until the agent leaves, so an
        // idling agent pays the -0.005/step loiter penalty instead of
        // oscillating 0.98->1.0 forever.
        if (!isCharging && !rechargeLocked && battery < 0.98f)
        {
            isCharging = true;
            wasChargeCycleThisVisit = true;
            batteryAtChargeStart = battery;
            rechargePartialPaid = false;
            rechargeFullPaid = false;
            if (chargeRewardMode == "none")
            {
                // No urgency gate — the policy learns WHEN to charge on its own.
                // Every charge earns the proportional reward; farming/camping is
                // prevented economically (kCharge*drainRate < step penalty) plus
                // lock-on-full and the escalating station idle penalty, none of
                // which encode when to charge.
                validChargeCycle = true;
                chargeCycleStarts++;
                minChargeStartBattery = Mathf.Min(minChargeStartBattery, batteryAtChargeStart);
            }
            else
            {
                validChargeCycle = IsRechargeUrgent();
                if (validChargeCycle)
                {
                    chargingStateFrozen = true;   // commit the obs to "charging" for this cycle
                    chargeCycleStarts++;
                    minChargeStartBattery = Mathf.Min(minChargeStartBattery, batteryAtChargeStart);
                }
            }
        }

        // Once charging, keep going until full.
        //  - "none" (V3): pay kCharge * (battery actually added) every step —
        //    dense, ungated, proportional. Charging-when-low is naturally worth
        //    more; the policy learns timing from the gradient, not a threshold.
        //  - "urgency" (V2): split bonus (small at 0.9, larger at full), valid
        //    cycles only; top-up cycles earn nothing.
        if (isCharging)
        {
            chargingSteps++;
            float batteryBefore = battery;
            battery = Mathf.Min(1.0f, battery + rechargeRatePerStep);
            float added = battery - batteryBefore;
            chargeProgress = Mathf.InverseLerp(rechargeThreshold, 1.0f, battery);

            if (chargeRewardMode == "none")
            {
                AddReward(kCharge * added);
            }
            else if (validChargeCycle && !rechargePartialPaid && battery >= rechargeBonusLevel)
            {
                rechargePartialPaid = true;
                AddReward(rechargePartialReward);
            }

            if (battery >= 1.0f)
            {
                chargeProgress = 1.0f;
                isCharging = false;
                // Lock on full regardless of mode: kills the top-up
                // oscillation (camping) loophole. No restart until exit.
                rechargeLocked = true;
                if (chargeRewardMode == "none")
                {
                    rechargeCount++;   // a completed full charge (already paid per-step)
                }
                else if (validChargeCycle && !rechargeFullPaid)
                {
                    rechargeFullPaid = true;
                    rechargeCount++;
                    AddReward(rechargeFullReward);
                }
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
                AddReward(-obstaclePenalty);
                renderGround.material.color = Color.red;
                break;

            case "timeout":
                if (trainingMode == TrainingMode.Task2)
                    AddReward(10f); 
                if (trainingMode == TrainingMode.Combined)  
                    AddReward((maxTargetsToReach - ptReachCount) * -10f);
                if (LearningBoard != null) LearningBoard.IncreaseTimeOut();
                if(trainingMode == TrainingMode.Task1)
                    AddReward(-50f);
                break;

            case "battery_dead":
                AddReward(-5f);
                // Same shortfall penalty as timeout, so dying is never a
                // cheaper exit than surviving without success
                if (trainingMode == TrainingMode.Combined)
                    AddReward((maxTargetsToReach - ptReachCount) * -10f);
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
            string minChargeStartLog = chargeCycleStarts > 0 ? $"{minChargeStartBattery:F3}" : "NA";
            string maxChargeExitLog = validChargeExits > 0 ? $"{maxChargeExitBattery:F3}" : "NA";
            string chargeGainedLog = validChargeExits > 0 ? $"{totalChargeGained:F3}" : "NA";
            string chargeEventsLog = chargeEventSummaries.Count > 0 ? string.Join(", ", chargeEventSummaries) : "NA";
            // Per-leg calibration data: completed-leg step counts and straight-line
            // distances (same length, index-aligned). Steps for leg i = ptReachSteps[i]
            // minus ptReachSteps[i-1] (leg 0 measured from step 0).
            string ptReachStepsLog = ptReachSteps.Count > 0 ? string.Join(",", ptReachSteps) : "NA";
            string ptLegDistsLog = ptLegDists.Count > 0 ? string.Join(",", ptLegDists.ConvertAll(d => d.ToString("F1"))) : "NA";
            string eventsLog;
            if (episodeEventTimeline.Count > 0)
            {
                var timeline = new List<string>(episodeEventTimeline)
                {
                    $"END@{episodeSteps}({reason},b={battery:F3})"
                };
                eventsLog = string.Join(" | ", timeline);
            }
            else
            {
                eventsLog = $"END@{episodeSteps}({reason},b={battery:F3})";
            }

            string episodeEndLog = $"[{timestamp}] [EPISODE END] reason={reason} | mode={trainingMode} | totalSteps={episodeSteps} | reward={GetCumulativeReward():F1} | ptReaches={ptReachCount} | bonusRecharges={rechargeCount} | batteryAtEnd={battery:F3} | batteryAtStart={batteryAtStart:F3} | batteryLife={batteryLifeSteps}steps | ptRate={ptRate:F2} | firstPtStep={firstPtReachStep} | chargeStarts={chargeCycleStarts} | topUps={topUpCount} | chargeExits={validChargeExits} | fullCharges={fullChargeCycles} | chargeGained={chargeGainedLog} | minChargeStart={minChargeStartLog} | maxChargeExit={maxChargeExitLog} | chargingSteps={chargingSteps} | stationIdleSteps={stationIdleSteps} | ptReachSteps=[{ptReachStepsLog}] | ptLegDists=[{ptLegDistsLog}] | chargeEvents={chargeEventsLog}";
            LearningBoard.WriteEpisodeLog(episodeEndLog);

            string episodeStatsLog = $"[{timestamp}] [EPISODE STATS] ptVisits={ptReachCount} | bonusRecharges={rechargeCount} | steps={episodeSteps} | mode={trainingMode} | avgPtVisits={LearningBoard.AvgPtVisits:F2} | avgBonusRecharges={LearningBoard.AvgRecharges:F2} | eps={LearningBoard.TotalEpisodes}";
            LearningBoard.WriteEpisodeLog(episodeStatsLog);
            LearningBoard.WriteEpisodeLog($"[{timestamp}] [EVENTS] {eventsLog}");
        }

        EndEpisode();
    }

    // ==================== FIXED UPDATE ====================

    void FixedUpdate()
    {
        episodeSteps++;
        AddPoints();
        UpdateBatteryLabel();
    }

    private void AddPoints()
    {
        if (segmentLines == null || activeSegment >= segmentLines.Length) return;
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

        while (!isValid && attempt < 300)
        {
            // Central box only: keeps worst-case agent->BS trip (corner to
            // far side of box ~506 units incl. detours) within the
            // post-threshold battery reserve, so charging is always reachable
            float rx = Random.Range(-bsSpawnRange, bsSpawnRange);
            float rz = Random.Range(-bsSpawnRange, bsSpawnRange);
            float fy = tfBatteryStation.localPosition.y;
            newPos = new Vector3(rx, fy, rz);

            isValid = true;
            foreach (Transform obstacle in obstaclesParent)
            {
                if (Vector3.Distance(newPos, obstacle.localPosition) < 40f)
                { isValid = false; break; }
            }
            // Don't stack on PT
            if (isValid && Vector3.Distance(newPos, tfTarget.localPosition) < 10f)
                isValid = false;
            // Don't stack on agent
            if (isValid && Vector3.Distance(newPos, tfAgent.localPosition) < 10f)
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
