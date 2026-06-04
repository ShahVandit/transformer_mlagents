using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors.Reflection;

public class SimpleTask1 : Agent
{
    public float VelMagnitude = 0;
    public bool VisualizeRay = true;
    private float gridSize = 500f;
    private Transform tfAgent;
    private Rigidbody rbAgent;
    private static System.DateTime fineTuningStartTime;
    private static bool fineTuningInitialized = false;
    // GridSensor static positioning
    private Transform gridSensorTransform;
    private Vector3 staticGridSensorPosition; // Will be calculated dynamically
    private Transform tfTarget;
    private Transform gridTarget;
    public float maxSpeed = 20f;
    private float addForce = 15.0f;
    private int contiguousUnvisitedCount = 0;
    public float rotation; //change wind direction
    public float windfactor;
    private float speedfactor = 1;

    private float distAfter;
    private float distBefore;

    private float orgdis;

    private Renderer renderGround;
    private Renderer renderTarget;

    private int trigger_frame = 20; // change here
    private int started_frame;
    private LearningDirector LearningBoard;
    public Vector3 windForce;

    public LineRenderer line;
    public Color lineColor;
    private int stationarySteps = 0;
    private Vector3 lastPosition;
    public int points;
    public Vector3 preposition;
    public Vector3 relaposition;
    public Vector3 position1;
    public int timesignal;
    public float coverage = 0;
    private float totalcoverage = 0;
    private int signalnumber = 0;
    private int lastGridX = -1;
    private int revisitCount = 0;
    private int episodeSteps = 0;
    private int lastGridZ = -1;
    private bool targetReached = false;
    private int ptReachedStep = -1;
    private int cellsAtPTReach = 0;
    private int gridCompletedStep = -1;
    private int ptCellX = 0;
    private int ptCellZ = 0;
    private int agentStartX = 0;
    private int agentStartZ = 0;
    private string firstTask = "n/a";
    //private Material gridTargetMaterial;
    //private Material TargetMaterial;
    private bool[,] gridVisited; // 5x5 grid for visited areas
    private Transform[] gridCells; // References to grid cells
    private Transform gridParent; // Reference to grid parent transform
    private List<Transform> allTargets = new List<Transform>();

    public override void Initialize()
    {
        //gridVisited = new bool[gridSize, gridSize];
        tfAgent = GetComponent<Transform>();
        rbAgent = GetComponent<Rigidbody>();
        tfTarget = transform.parent.Find("Target").gameObject.GetComponent<Transform>();
        timesignal = 1;
        points = 1;
        if (!fineTuningInitialized)
        {
            fineTuningStartTime = System.DateTime.Now;
            fineTuningInitialized = true;
            Debug.Log($"[CURRICULUM] Fine-tuning started at {fineTuningStartTime}");
        }
        preposition = tfTarget.localPosition;
        position1 = tfAgent.localPosition;

        renderGround = transform.parent.Find("Ground").gameObject.GetComponent<Renderer>();
        renderTarget = tfTarget.gameObject.GetComponent<Renderer>();
        //targetPositions.Clear();

        // Initialize grid visited matrix
        gridVisited = new bool[5, 5];

        // Load all grid cells (assuming they're children of the "Grid" object)
        gridParent = transform.parent.Find("Grid");
        if (gridParent != null)
        {
            gridCells = new Transform[25];
            for (int i = 0; i < gridParent.childCount; i++)
            {
                gridCells[i] = gridParent.GetChild(i);
            }

            // Calculate GridSensor position
            // GridSensor's origin is at its CENTER, so we need to position it at the southwest corner of the grid
            // Grid starts at gridParent.position, so GridSensor goes there (no offset needed)
            staticGridSensorPosition = gridParent.position;
        }
        else
        {
            Debug.LogError("Grid parent object not found.");
        }

        // Find GridSensorHolder child object
        Transform gridSensorHolder = transform.Find("GridSensorHolder");
        if (gridSensorHolder != null)
        {
            gridSensorTransform = gridSensorHolder;
        }
        else
        {
            Debug.LogWarning("GridSensorHolder not found on Drone Agent - this is OK if not using GridSensor");
        }
    }
    void Start()
    {
        LearningBoard = GameObject.Find("LearningDirector")?.GetComponent<LearningDirector>();
        if (LearningBoard == null)
        {
            Debug.LogError("LearningDirector not found or missing LearningDirector component.");
        }
        //gridTargetMaterial = Resources.Load<Material>("Materials/Grid Target");
        //TargetMaterial = Resources.Load<Material>("Materials/Blue Target");
    }

    public override void OnEpisodeBegin()
    {
        rbAgent.velocity = Vector3.zero;
        rbAgent.angularVelocity = Vector3.zero;
        tfAgent.eulerAngles = Vector3.zero;
        relaposition = Vector3.zero;
        line.positionCount = 0;
        points = 1;
        revisitCount= 0;
        MaxStep = 80000;
        ResetGridColors(); // Reset grid colors at the start of each episode
        RandomizeObstaclePositionsAndSizes(); // Randomize obstacles
        RandomizeTargetPosition();
        RandomizeGridTargetPositions();
        RandomizeAgentPosition();
        episodeSteps=0;
        ptReachedStep = -1;
        cellsAtPTReach = 0;
        gridCompletedStep = -1;
        firstTask = "n/a";
        // Reactivate all grid targets
        //foreach (Transform target in allTargets)
        //{
        //    Renderer targetRenderer = tfTarget.GetComponent<Renderer>();
        //    targetRenderer.material.color = originalColor;
        //    target.gameObject.SetActive(true);
        //}

        this.gameObject.SetActive(true);
        foreach (Transform target in allTargets)
        {
            target.gameObject.SetActive(true);
        }
        MarkStartingCell();
        
        gridTarget = GetNearestGridTarget();
        // tfTarget.gameObject.SetActive(true);
        // PT target stays active — reachable throughout episode
        //Debug.Log($"Total Grid Targets Found: {allTargets.Count}");
        // Set the first target as the current target
        //tfTarget = GetNearestGridTarget();
        //if (tfTarget != null)
        //{
        //    Renderer targetRenderer = tfTarget.GetComponent<Renderer>();
        //    targetRenderer.material.color = Color.magenta; // Highlight the current target
        //    orgdis = Vector3.Distance(tfAgent.position, tfTarget.position);
        //    distBefore = orgdis;
        //}
        //else
        //{
        //    Debug.LogError("No targets found.");
        //}
        distBefore = Vector3.Distance(tfAgent.position, tfTarget.position);
        StartCoroutine(RevertMaterial());
    }

    private void MarkStartingCell()
    {
        float mapSize = 500f;
        float cellSize = mapSize / 5;
        float halfMap = mapSize / 2;

        Vector3 agentPos = tfAgent.localPosition;
        int gridX = Mathf.Clamp((int)((agentPos.x + halfMap) / cellSize), 0, 4);
        int gridZ = Mathf.Clamp((int)((agentPos.z + halfMap) / cellSize), 0, 4);

        lastGridX = gridX;
        lastGridZ = gridZ;

        if (!gridVisited[gridX, gridZ])
        {
            gridVisited[gridX, gridZ] = true;
            contiguousUnvisitedCount = 1;

            int gridIndex = gridZ * 5 + gridX;
            if (gridCells[gridIndex] != null)
            {
                var renderer = gridCells[gridIndex].GetComponent<Renderer>();
                if (renderer != null)
                    renderer.material.color = Color.green;
            }
            Transform startTarget = tfTarget.parent.Find($"Target ({gridIndex})");
            if (startTarget != null)
                GridTargetReached(startTarget);
        }
    }
    private void RandomizeAgentPosition()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        Transform agentTransform = this.transform;

        if (obstaclesParent != null && agentTransform != null)
        {
            bool isPositionValid = false;
            Vector3 newAgentPosition = Vector3.zero;

            int maxAttempts = 100; // Prevent infinite loops
            int attempt = 0;

            while (!isPositionValid && attempt < maxAttempts)
            {
                // Generate a random position for the agent
                float randomX = Random.Range(-200f, 200f);
                float randomZ = Random.Range(-200f, 200f);
                float fixedY = 20;

                newAgentPosition = new Vector3(randomX, fixedY, randomZ);

                // Check for conflicts with obstacles
                isPositionValid = true;
                foreach (Transform obstacle in obstaclesParent)
                {
                    if (Vector3.Distance(newAgentPosition, obstacle.localPosition) < 50f) // Adjust minimum distance as needed
                    {
                        isPositionValid = false;
                        break;
                    }
                }

                attempt++;
            }

            if (isPositionValid)
            {
                agentTransform.localPosition = newAgentPosition;
                float agentHalfMap = 250f;
                float agentCellSize = 100f;
                agentStartX = Mathf.Clamp((int)((newAgentPosition.x + agentHalfMap) / agentCellSize), 0, 4);
                agentStartZ = Mathf.Clamp((int)((newAgentPosition.z + agentHalfMap) / agentCellSize), 0, 4);
            }
            else
            {
                Debug.LogError("Failed to find a valid position for the agent after maximum attempts.");
            }
        }
        else
        {
            Debug.LogError("Obstacles or Agent object not found.");
        }
    }
    private void RandomizeGridTargetPositions()
    {
        allTargets.Clear();
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        gridParent = transform.parent.Find("Grid");

        if (obstaclesParent != null && gridParent != null)
        {
            // Define the original color using RGBA values
            Color originalColor = new Color(255f / 255f, 234f / 255f, 0f / 255f, 255f / 255f);

            // Loop through all targets (0-24)
            for (int i = 0; i < 25; i++)
            {
                Transform targetTransform = transform.parent.Find($"Target ({i})");
                if (targetTransform != null)
                {
                    // Find corresponding grid cell
                    Transform gridCell = gridCells[i];
                    if (gridCell != null)
                    {
                        Vector3 cellPosition = gridCell.position;
                        Vector3 cellScale = gridCell.localScale;

                        // Calculate world position

                        // Check for obstacles

                        //attempt++;
                        //if (isPositionValid)
                        //{
                        //    targetTransform.position = newTargetPosition;
                        //    targetTransform.gameObject.SetActive(true);
                        //    allTargets.Add(targetTransform);

                        //    // Assign the original color to the target
                        //    Renderer targetRenderer = targetTransform.GetComponent<Renderer>();
                        //    if (targetRenderer != null)
                        //    {
                        //        targetRenderer.material.color = originalColor;
                        //    }
                        //}
                        //else
                        //{

                        // Fallback to cell center if no valid position found
                        targetTransform.position = cellPosition;
                        targetTransform.gameObject.SetActive(true);
                        allTargets.Add(targetTransform);
                        Renderer targetRenderer = targetTransform.GetComponent<Renderer>();
                        if (targetRenderer != null)
                        {
                            targetRenderer.material.color = originalColor;
                        }


                        //}
                    }
                    else
                    {
                        Debug.LogError($"Grid cell {i} not found");
                    }
                }
                else
                {
                    Debug.LogError($"Target ({i}) not found");
                }
            }
        }
        else
        {
            Debug.LogError("Obstacles or Grid parent object not found.");
        }


    }

    private void RandomizeObstaclePositionsAndSizes()
    {
        Transform obstaclesParent = transform.parent.Find("Obstacles");
        if (obstaclesParent != null)
        {
            int gridSize = 5; // A 5x5 grid for 25 obstacles
            float cellSize = 500f / gridSize; // Size of each grid cell

            int obstacleIndex = 0;
            foreach (Transform obstacle in obstaclesParent)
            {
                // Calculate grid row and column
                int row = obstacleIndex / gridSize;
                int col = obstacleIndex % gridSize;

                // Determine the center of the current grid cell
                float baseX = -250f + col * cellSize + cellSize / 2;
                float baseZ = -250f + row * cellSize + cellSize / 2;

                // Add small random offsets to make positions less uniform
                float randomOffsetX = Random.Range(-cellSize / 4, cellSize / 4);
                float randomOffsetZ = Random.Range(-cellSize / 4, cellSize / 4);
                float fixedY = obstacle.localPosition.y; // Keep Y constant

                // Assign the new random position
                obstacle.localPosition = new Vector3(baseX + randomOffsetX, fixedY, baseZ + randomOffsetZ);

                // Define the range for random sizes
                float randomScaleX = Random.Range(30f, 50f);
                float randomScaleZ = Random.Range(30f, 50f);

                // Assign the new random scale
                obstacle.localScale = new Vector3(randomScaleX, 100, randomScaleZ);

                obstacleIndex++;
                if (obstacleIndex >= gridSize * gridSize)
                    break; // Stop if we have placed all obstacles
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

            int maxAttempts = 100; // Prevent infinite loops
            int attempt = 0;

            while (!isPositionValid && attempt < maxAttempts)
            {
                // Generate a random position for the target
                float randomX = Random.Range(-200f, 200f);
                float randomZ = Random.Range(-200f, 200f);
                float fixedY = targetTransform.localPosition.y; // Keep Y constant

                newTargetPosition = new Vector3(randomX, fixedY, randomZ);

                // Check for conflicts with obstacles
                isPositionValid = true;
                foreach (Transform obstacle in obstaclesParent)
                {
                    if (Vector3.Distance(newTargetPosition, obstacle.localPosition) < 50f) // Minimum safe distance
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
                float ptHalfMap = 250f;
                float ptCellSize = 100f;
                ptCellX = Mathf.Clamp((int)((newTargetPosition.x + ptHalfMap) / ptCellSize), 0, 4);
                ptCellZ = Mathf.Clamp((int)((newTargetPosition.z + ptHalfMap) / ptCellSize), 0, 4);
                //targetPositions.Enqueue(new Tuple<Vector3, string>(newTargetPosition, $"Target"));
                //targetPositions.Enqueue((newTargetPosition, $"Target"));

                tfTarget = transform.parent.Find("Target").gameObject.GetComponent<Transform>();
                tfTarget.gameObject.SetActive(true);
                Renderer targetRenderer = tfTarget.GetComponent<Renderer>();

            }
            else
            {
                Debug.LogError("Failed to find a valid position for the target after maximum attempts.");
            }
        }
        else
        {
            Debug.LogError("Obstacles or Target object not found.");
        }
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float mapSize = 500f;
        float cellSize = mapSize / 5f;
        float halfMap = mapSize / 2f;

        Vector3 agentPos = tfAgent.localPosition;
        int currentRow = Mathf.Clamp((int)((agentPos.x + halfMap) / cellSize), 0, 4);
        int currentCol = Mathf.Clamp((int)((agentPos.z + halfMap) / cellSize), 0, 4);

        // ========== OPTION B: 90-DIM OBSERVATIONS WITH EXPLICIT SPATIAL ENCODING ==========
        // This encoding makes spatial relationships EXPLICIT so the agent knows:
        // - Which cells are neighbors (Manhattan distance = 1)
        // - Where each cell is relative to the agent
        // - Grid structure is NOT implicit - it's directly encoded!

        // === 1. VELOCITY (2 dims) ===
        sensor.AddObservation(rbAgent.velocity.x / maxSpeed);
        sensor.AddObservation(rbAgent.velocity.z / maxSpeed);

        // === 2. PROGRESS (1 dim) ===
        int totalVisited = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (gridVisited[i, j]) totalVisited++;
        sensor.AddObservation(totalVisited / 25f);

        // === 3. TARGET DIRECTION (2 dims) ===
        if (tfTarget != null)
        {
            Vector3 relativePos = tfTarget.localPosition - tfAgent.localPosition;
            sensor.AddObservation(relativePos.x / 500f);
            sensor.AddObservation(relativePos.z / 500f);
        }
        else
        {
            sensor.AddObservation(0f);
            sensor.AddObservation(0f);
        }

        // === 4. SPATIALLY-ENCODED GRID STATE (75 dims = 25 cells × 3) ===
        // For each cell: [visited_flag, deltaRow, deltaCol]
        // This makes adjacency EXPLICIT: neighbors have |deltaRow| + |deltaCol| = 1
        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                // Visited flag (0 or 1)
                sensor.AddObservation(gridVisited[row, col] ? 1f : 0f);

                // Relative position from current cell (ego-centric encoding)
                int deltaRow = row - currentRow;  // Range: -4 to +4
                int deltaCol = col - currentCol;  // Range: -4 to +4

                sensor.AddObservation(deltaRow / 4f);  // Normalized: -1.0 to +1.0
                sensor.AddObservation(deltaCol / 4f);  // Normalized: -1.0 to +1.0

                // Example: If agent is at [2,2] and cell is [2,3]:
                //   deltaRow = 0, deltaCol = +1 → (0.0, 0.25) → Adjacent neighbor above
                // Example: If agent is at [0,0] and cell is [4,4]:
                //   deltaRow = +4, deltaCol = +4 → (1.0, 1.0) → Far diagonal corner
            }
        }

        // === 5. CURRENT CELL INDEX (2 dims) - EXPLICIT GRID LOCATION ===
        sensor.AddObservation(currentRow / 4f);  // 0.0, 0.25, 0.5, 0.75, 1.0
        sensor.AddObservation(currentCol / 4f);  // 0.0, 0.25, 0.5, 0.75, 1.0

        // === 6. DISTANCE TO ALL 4 CELL BOUNDARIES (4 dims) - BOUNDARY AWARENESS ===
        float cellMinX = -halfMap + currentRow * cellSize;
        float cellMaxX = cellMinX + cellSize;
        float cellMinZ = -halfMap + currentCol * cellSize;
        float cellMaxZ = cellMinZ + cellSize;

        // Absolute distances to each boundary (normalized to 0-1 range within cell)
        float distToLeft = (agentPos.x - cellMinX) / cellSize;
        float distToRight = (cellMaxX - agentPos.x) / cellSize;
        float distToBottom = (agentPos.z - cellMinZ) / cellSize;
        float distToTop = (cellMaxZ - agentPos.z) / cellSize;

        sensor.AddObservation(distToLeft);    // Distance to LEFT boundary (0 at edge, 1 at right)
        sensor.AddObservation(distToRight);   // Distance to RIGHT boundary (0 at edge, 1 at left)
        sensor.AddObservation(distToBottom);  // Distance to BOTTOM boundary (0 at edge, 1 at top)
        sensor.AddObservation(distToTop);     // Distance to TOP boundary (0 at edge, 1 at bottom)

        // === 7. NEIGHBOR VISITATION STATUS (4 dims) - IMMEDIATE CONTEXT ===
        // Explicit adjacency flags for immediate decision-making
        sensor.AddObservation((currentRow > 0 && gridVisited[currentRow-1, currentCol]) ? 1f : 0f);  // Left (-X)
        sensor.AddObservation((currentRow < 4 && gridVisited[currentRow+1, currentCol]) ? 1f : 0f);  // Right (+X)
        sensor.AddObservation((currentCol > 0 && gridVisited[currentRow, currentCol-1]) ? 1f : 0f);  // Down (-Z)
        sensor.AddObservation((currentCol < 4 && gridVisited[currentRow, currentCol+1]) ? 1f : 0f);  // Up (+Z)

        // ========== TOTAL: 2 + 1 + 2 + 75 + 2 + 4 + 4 = 90 dimensions ==========
    }



    public override void OnActionReceived(ActionBuffers actions)
    {
        //// Calculate distance to target
        // if (tfTarget.gameObject.activeSelf)
        // {
        //     float distanceToTarget = Vector3.Distance(tfAgent.position, tfTarget.position);
        //     float distanceReward = (distBefore - distanceToTarget) * 0.5f;
        //     distBefore = distanceToTarget;

        //     AddReward(distanceReward);
        // }
        //else
        //{
        // float distanceToGridTarget = Vector3.Distance(tfAgent.position, gridTarget.position);
        // float gridDistanceReward = (distBefore - distanceToGridTarget);
        // distBefore = distanceToGridTarget;
        // AddReward(gridDistanceReward);
        //    //}
           // Small step penalty to encourage efficiency
           AddReward(-0.001f);
        //    //}
           // Execute movement actions
        //}
        float moveX = actions.ContinuousActions[0];
        float moveZ = actions.ContinuousActions[1];
        Vector3 moveDirection = new Vector3(moveX, 0, moveZ);
        if (moveDirection.magnitude > 0.01f)  // Avoid normalizing zero vector
        {
            moveDirection = moveDirection.normalized;
        }

        rbAgent.velocity = moveDirection * maxSpeed;

        Vector3 currentPosition = tfAgent.position;
        lastPosition = currentPosition;

        // Debug boundary distances (every 100 steps)
        // if (this.StepCount % 100 == 0)
        // {
        //     float mapSize = 500f;
        //     float cellSize = mapSize / 5;
        //     float halfMap = mapSize / 2;
        //     Vector3 agentPos = tfAgent.localPosition;
        //     int currentRow = Mathf.Clamp((int)((agentPos.x + halfMap) / cellSize), 0, 4);
        //     int currentCol = Mathf.Clamp((int)((agentPos.z + halfMap) / cellSize), 0, 4);

        //     float cellMinX = -halfMap + currentRow * cellSize;
        //     float cellMaxX = cellMinX + cellSize;
        //     float cellMinZ = -halfMap + currentCol * cellSize;
        //     float cellMaxZ = cellMinZ + cellSize;

        //     float distToLeft = (agentPos.x - cellMinX) / cellSize;
        //     float distToRight = (cellMaxX - agentPos.x) / cellSize;
        //     float distToBottom = (agentPos.z - cellMinZ) / cellSize;
        //     float distToTop = (cellMaxZ - agentPos.z) / cellSize;

        //     Debug.Log($"[BOUNDARY] Cell({currentRow},{currentCol}) Pos({agentPos.x:F1},{agentPos.z:F1}) " +
        //               $"Bounds[{cellMinX:F1},{cellMaxX:F1}]x[{cellMinZ:F1},{cellMaxZ:F1}] " +
        //               $"Dist[L:{distToLeft:F2} R:{distToRight:F2} B:{distToBottom:F2} T:{distToTop:F2}] " +
        //               $"Sum[X:{(distToLeft+distToRight):F2} Z:{(distToBottom+distToTop):F2}]");
        // }

        // If max steps reached, penalize and reset episode
        if (this.StepCount >= MaxStep)
        {
            // Debug.Log("Max step reached");  // Commented out - too verbose
            EndEpisodeCustom("timeout", -45f);
        }

        // Verbose logging commented out
        // string gridTargetName = gridTarget != null ? gridTarget.name : "null";
        // int gridTargetIndex = -1;
        // if (gridTarget != null && gridTargetName.Contains("(") && gridTargetName.Contains(")"))
        // {
        //     int startIdx = gridTargetName.IndexOf("(") + 1;
        //     int endIdx = gridTargetName.IndexOf(")");
        //     if (int.TryParse(gridTargetName.Substring(startIdx, endIdx - startIdx), out gridTargetIndex))
        //     {
                // Debug.Log($" Cumulative Reward: {GetCumulativeReward()}");
        //     }
        // }

        // Milestone logging kept
        // if (Academy.Instance.StepCount % 50000 == 0)
        // {
        //     Debug.Log($"[MILESTONE] Global Step: {Academy.Instance.StepCount}");
        // }
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

    private void OnCollisionEnter(Collision collision)
    {
        if (collision.gameObject.tag.Equals("Target"))
        {

            TargetReached();
        }
        if (collision.gameObject.tag.Equals("gridTarget"))
        {
        //    GridTargetReached(collision.gameObject.transform);
        }
        else if (collision.gameObject.tag.Equals("Geometry"))
        {
            
            timesignal = 0;
            EndEpisodeCustom("obstacle", -70f);
        }
    }

    IEnumerator RevertMaterial()
    {
        yield return new WaitForSeconds(0.01f);
        renderGround.material.color = Color.white;
        renderTarget.material.color = Color.blue;
    }


    void FixedUpdate()
    {
        episodeSteps++;
        UpdateVisitedGrid();
        AddPoints();
    }

    void LateUpdate()
    {
        // Keep GridSensor statically positioned at grid center
        if (gridSensorTransform != null)
        {
            gridSensorTransform.position = staticGridSensorPosition;
            gridSensorTransform.rotation = Quaternion.identity; // Keep it flat
        }
    }

    public void AddPoints()
    {
        Vector3 pt = transform.position;

        if (points == 1)
        {
            line.positionCount = points;
            line.SetPosition(points - 1, pt);
            points++;
        }
        else if (Vector3.Distance(line.GetPosition(points - 2), pt) >= 0.5f)
        {
            if (pt != Vector3.zero)
            {
                line.positionCount = points;
                line.SetPosition(points - 1, pt);
                points++;
            }
        }
    }




    private Transform GetNearestGridTarget()
    {
        Transform closestTarget = null;
        float minDistance = Mathf.Infinity;

        foreach (Transform target in allTargets)
        {
            if (target.gameObject.activeSelf) // Only consider active targets
            {
                float distance = Vector3.Distance(tfAgent.position, target.position);
                if (distance < minDistance)
                {
                    minDistance = distance;
                    closestTarget = target;
                }
            }
        }

        return closestTarget;
    }
    private void UpdateVisitedGrid()
    {
        float mapSize = 500f;
        float cellSize = mapSize / 5;
        float halfMap = mapSize / 2;

        Vector3 agentPos = tfAgent.localPosition;
        int gridX = Mathf.Clamp((int)((agentPos.x + halfMap) / cellSize), 0, 4);
        int gridZ = Mathf.Clamp((int)((agentPos.z + halfMap) / cellSize), 0, 4);

        // Only run logic if the agent has entered a NEW cell
        if (gridX != lastGridX || gridZ != lastGridZ)
        {
            if (gridVisited[gridX, gridZ])
            {
                // Only trigger ONCE when stepping into a visited cell again
                contiguousUnvisitedCount = 0;
                HandleRevisit(gridX, gridZ);
            }
            else
            {
                contiguousUnvisitedCount++;
                gridVisited[gridX, gridZ] = true;
                int gridIndex = gridZ * 5 + gridX;
                if (gridCells[gridIndex] != null)
                {
                    GridTargetReached(tfTarget.parent.Find($"Target ({gridIndex})"));
                    var renderer = gridCells[gridIndex].GetComponent<Renderer>();
                    if (renderer != null)
                        renderer.material.color = Color.green;
                }
            }

            // Update last visited coordinates
            lastGridX = gridX;
            lastGridZ = gridZ;
        }

        CheckAllGridVisited();
    }

    private float GetRevisitPenaltyScale()
    {
        System.TimeSpan elapsed = System.DateTime.Now - fineTuningStartTime;
        double hoursElapsed = elapsed.TotalHours;
        
        if (hoursElapsed < 2.5) return 1.0f;       // -5f
        else if (hoursElapsed < 5.0) return 1.2f;  // -6f
        else return 1.2f;                          // CAP at -6f permanently
    }
    private void HandleRevisit(int gridX, int gridZ)
    {
        revisitCount++;
        float scale = GetRevisitPenaltyScale();
    }
    private void CheckAllGridVisited()
    {
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                if (!gridVisited[i, j])
                {
                    // If any cell is not visited, return without ending the episode
                    return;
                }
            }
        }
        // EpisodeSuccess();
        //Debug.Log("Reward full");
        // If all cells are visited, the episode is successful
        renderGround.material.color = Color.green;
        // Set Grid_first as soon as all cells are done, if PT not yet reached
        if (firstTask == "n/a" && ptReachedStep < 0)
            firstTask = "Grid_first";
        if (!tfTarget.gameObject.activeSelf)
        {
            if (gridCompletedStep < 0)
                gridCompletedStep = episodeSteps;
            EpisodeSuccess();
        }

    }
    private void GridTargetReached(Transform collidedTarget)
    {

        collidedTarget.gameObject.SetActive(false);
        Transform nextTarget = GetNearestGridTarget();
        if (nextTarget != null)
        {
            gridTarget = nextTarget;
            if (!tfTarget.gameObject.activeSelf)
            {
                distBefore = Vector3.Distance(tfAgent.position, gridTarget.position); // Resetting the distance
            }
            Renderer targetRenderer = gridTarget.GetComponent<Renderer>();
            targetRenderer.material.color = Color.magenta;
        }
        // Balanced: Progressive reward for visiting new cells
        float reward = 10f;
        AddReward(reward);
    }
    public void TargetReached()
    {
        ptReachedStep = episodeSteps;
        int ptCnt = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (gridVisited[i, j]) ptCnt++;
        cellsAtPTReach = ptCnt;
        if (firstTask == "n/a") firstTask = "PT_first";

        renderGround.material.color = Color.green;
        renderTarget.material.color = Color.green;
        timesignal = 0;
        distBefore = Vector3.Distance(tfAgent.position, gridTarget.position); // Resetting the distance
        tfTarget.gameObject.SetActive(false);
        // EndEpisodeCustom("success", 100f);
        AddReward(50f);
        CheckAllGridVisited();

    }
    private void EpisodeSuccess()
    {
        if (revisitCount == 0)
        {
            EndEpisodeCustom("perfect_success", 150f);
        }
        else
        {
            EndEpisodeCustom("success", 100f);
        }
    }
    private void EndEpisodeCustom(string reason, float reward)
    {
        // AddReward(reward);
        int cellsAtEnd = 0;
        for (int i = 0; i < 5; i++)
            for (int j = 0; j < 5; j++)
                if (gridVisited[i, j]) cellsAtEnd++;

        string ptField = ptReachedStep >= 0 ? $"step {ptReachedStep} ({cellsAtPTReach}/25 cells)" : "incomplete";
        string gcField = gridCompletedStep >= 0 ? $"step {gridCompletedStep}" : "incomplete";
        string gapField = "n/a";
        if (ptReachedStep >= 0 && gridCompletedStep >= 0)
            gapField = Mathf.Abs(gridCompletedStep - ptReachedStep).ToString();

        string timestamp = System.DateTime.Now.ToString("yyyy-MM-dd HH:mm:ss");
        string episodeLog = $"[{timestamp}] [EPISODE END] reason={reason} | totalSteps={episodeSteps} | reward={GetCumulativeReward():F1} | PointTarget={ptField} | GridCover={gcField} | First={firstTask} | Gap={gapField} | Revisits={revisitCount} | CellsAtEnd={cellsAtEnd} | PTCell={ptCellX},{ptCellZ} | AgentStart={agentStartX},{agentStartZ}";
        Debug.Log(episodeLog);
        LearningBoard.WriteEpisodeLog(episodeLog);
        LearningBoard.IncreaseRevisit(revisitCount);
        // if(revisitCount>0){
        //     Debug.Log(revisitCount);
        // } 
        switch (reason)
        {
            case "obstacle":
                LearningBoard.IncreaseFailed();
                AddReward(-40f);  // Balanced: Strong penalty but recoverable
                renderGround.material.color = Color.red;
                break;

            case "timeout":
                AddReward(-50f);  // Balanced: Moderate penalty for running out of time
                LearningBoard.IncreaseTimeOut();
                break;

            case "success":
                // Scale smoothly with revisit count
                float baseReward = 200f;
                float revisitPenalty = revisitCount * 15f;
                AddReward(150f);
                // 0 revisits = 200f
                // 3 revisits = 155f
                // 10 revisits = 50f (floor)
                LearningBoard.IncreaseSuccess();
                break;

            case "perfect_success":
                AddReward(150f);  // Down from 800f
                LearningBoard.IncreasePerfect();
                LearningBoard.IncreaseSuccess();
                break;

            default:
                Debug.LogWarning("EndEpisodeCustom called with unknown reason");
                break;
        }
        // End the episode after assigning the reward
        EndEpisode();
    }
    private void ResetGridColors()
    {
        // Reset the gridVisited matrix
        for (int row = 0; row < 5; row++)
        {
            for (int col = 0; col < 5; col++)
            {
                gridVisited[row, col] = false;
            }
        }

        // Reset the color of all grid cells to the default (e.g., white)
        if (gridCells != null)
        {
            foreach (Transform gridCell in gridCells)
            {
                if (gridCell != null)
                {
                    Renderer renderer = gridCell.GetComponent<Renderer>();
                    if (renderer != null)
                    {
                        renderer.material.color = Color.gray; // Default grid color
                    }
                }
            }
        }
    }

}
