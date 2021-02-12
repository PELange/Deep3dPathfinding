using Accord.MachineLearning;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using UnityEngine;

namespace Assets.Scripts.VoxelWorld
{
    public class VoxelWorldGenerator
    {
        public class Parameters
        {
            public int Width { get; set; }
            public int Height { get; set; }
            public int Depth { get; set; }

            public LevelType levelType;

            public int Seed { get; set; }
        }
        public enum LevelType
        {
            Noise = 0,
            Canyon = 1,
            Maze = 2,
            Hills = 3,
            smallHills = 4

        }
        public Parameters GenerationParameters { get; set; }

        public System.Random Random { get; set; }

        public VoxelWorld World { get; set; }

        public List<List<VoxelSegment>> SegmentLayers { get; set; }

        public List<VoxelElevator> Elevators { get; set; }

        public int[,,] rLLevel;
        public Vector3Int Startpos { get; set; }

        public Vector3Int Endpos { get; set; }

        public List<Vector3Int> Path { get; set; }

        int freevoxels, blockedvoxels;
        public int blockedPercentage;

        public static VoxelWorldGenerator Run(Parameters parameters)
        {
            var generator = new VoxelWorldGenerator
            {
                GenerationParameters = parameters,
                Random = new System.Random(parameters.Seed),
                World = new VoxelWorld(parameters.Width, parameters.Height, parameters.Depth, parameters.levelType),
                SegmentLayers = new List<List<VoxelSegment>>(),
                Elevators = new List<VoxelElevator>()
            };

            //1. Fill the world with voxels based on 3D-Perlin noise
            generator.PopulateVoxels();

            //2. Group together segments
            generator.FindSegmentLayers();

            //3. Place elevators
            generator.PopulateElevators();



            return generator;
        }

        protected void PopulateVoxels()
        {
            freevoxels = 0;
            blockedvoxels = 0;
            for (int height = 0; height < World.Height; height++)
            {

                for (int z = 0; z < World.Depth; z++)
                {
                    for (int x = 0; x < World.Width; x++)
                    {
                        //bottom layer -> solid type
                        if (height == 0)
                        {
                            World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                            continue;
                        }

                        //sky layer -> empty
                        if (height == World.Height - 1 && World.Height > 2)
                        {
                            World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                            continue;
                        }


                        float noiseValue;
                        switch (World.levelType)
                        {
                            case LevelType.Noise:
                            {
                                noiseValue = Mathf.PerlinNoise(x / 6f + height * 10, z / 6f + height * 10 + GenerationParameters.Seed);
                                if (noiseValue > 0.5)
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                                    blockedvoxels++;
                                }
                                else
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                                    freevoxels++;
                                }
                                break;
                            }
                            case LevelType.Canyon:
                            {
                                noiseValue = Mathf.PerlinNoise(x / 3.2f + GenerationParameters.Seed, z / 3.1f + GenerationParameters.Seed);
                                noiseValue = Mathf.Lerp(noiseValue * 1.2f, 0.4f, (float)height / (float)World.Height);
                                if (noiseValue > 0.5f)
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                                    blockedvoxels++;
                                }
                                else
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                                    freevoxels++;
                                }
                                break;
                            }
                            case LevelType.Maze:
                            {
                                noiseValue = Mathf.PerlinNoise(x * 5.2f + GenerationParameters.Seed, z * 4.1f + GenerationParameters.Seed);

                                if (noiseValue > 0.6)
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                                    blockedvoxels++;
                                }
                                else
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                                    freevoxels++;
                                }
                                break;
                            }
                            case LevelType.Hills:
                            {
                                float scale = 1.98f / World.Width;

                                float heightscale = (float)height / (float)(World.Height - 1);
                                noiseValue = Mathf.PerlinNoise(x * scale + GenerationParameters.Seed, z * scale + GenerationParameters.Seed);

                                //noiseValue = Mathf.Lerp(noiseValue*25.5f+0.3f,-3.1f, heightscale);
                                if (World.Height != 2)
                                    noiseValue = (noiseValue * (1.0f - 0.5f * (heightscale * heightscale)));

                                if (noiseValue > 0.5f)
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                                    blockedvoxels++;
                                }
                                else
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                                    freevoxels++;
                                }
                                break;
                            }

                            case LevelType.smallHills:
                            {
                                float scale = 10.98f / World.Width;

                                float heightscale = (float)height / (float)(World.Height - 1);
                                noiseValue = Mathf.PerlinNoise(x * scale + GenerationParameters.Seed, z * scale + GenerationParameters.Seed);

                                //noiseValue = Mathf.Lerp(noiseValue*25.5f+0.3f,-3.1f, heightscale);
                                if (World.Height != 2)
                                    noiseValue = (noiseValue * (1.0f - 0.55f * (heightscale * heightscale)));

                                if (noiseValue > 0.61f)
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Solid));
                                    blockedvoxels++;
                                }
                                else
                                {
                                    World.TrySetVoxel(new Voxel(x, height, z, Voxel.VoxelType.Empty));
                                    freevoxels++;
                                }
                                break;
                            }
                        }

                    }
                }
            }
            blockedPercentage = (int)((float)blockedvoxels / (float)(freevoxels + blockedvoxels) * 100f);

        }

        protected void FindSegmentLayers()
        {
            int nextLabel = 2;
            var labels = new int[World.Width, World.Height, World.Depth];

            //Seperate background and foreground for each layer
            for (int y = 0; y < World.Height; y++)
            {
                for (int z = 0; z < World.Depth; z++)
                {
                    for (int x = 0; x < World.Width; x++)
                    {

                        if (World.TryGetVoxel(x, y, z, out var voxel) && voxel.Type != Voxel.VoxelType.Empty) //Found a block that is not empty ...
                        {
                            labels[x, y, z] = 1; //Set label to 1 = something to be labelled. The default value 0 means voxel can not have a label
                        }
                    }
                }
            }

            //Perform floodfill for each layer
            for (int y = 0; y < World.Height; y++)
            {

                for (int z = 0; z < World.Depth; z++)
                {
                    for (int x = 0; x < World.Width; x++)
                    {
                        if (labels[x, y, z] == 1 && World.TryGetVoxel(x, y, z, out var labelSegmentStart)) //Found a block that still needs a label
                        {
                            var stack = new Stack<Voxel>();

                            stack.Push(labelSegmentStart);

                            while (stack.Count > 0)
                            {
                                var voxel = stack.Pop();

                                if (labels[voxel.Position.X, voxel.Position.Y, voxel.Position.Z] == 1)
                                {
                                    labels[voxel.Position.X, voxel.Position.Y, voxel.Position.Z] = nextLabel; //Assign current label

                                    //Enqueue neighbours on the same layer (y)
                                    if (World.TryGetVoxel(voxel.Position.X - 1, voxel.Position.Y, voxel.Position.Z, out var left)) stack.Push(left);
                                    if (World.TryGetVoxel(voxel.Position.X + 1, voxel.Position.Y, voxel.Position.Z, out var right)) stack.Push(right);
                                    if (World.TryGetVoxel(voxel.Position.X, voxel.Position.Y, voxel.Position.Z - 1, out var front)) stack.Push(front);
                                    if (World.TryGetVoxel(voxel.Position.X, voxel.Position.Y, voxel.Position.Z + 1, out var back)) stack.Push(back);
                                }
                            }

                            nextLabel++;
                        }
                    }
                }
            }

            //Find the segments for each layer
            for (int y = 0; y < World.Height; y++)
            {
                var currentSegmentLayer = new List<VoxelSegment>();


                for (int z = 0; z < World.Depth; z++)
                {
                    for (int x = 0; x < World.Width; x++)
                    {
                        var label = labels[x, y, z];

                        if (label < 2) continue; //Ignore air and 

                        var segment = currentSegmentLayer.Where(element => element.Label == label).FirstOrDefault();

                        if (segment == null)
                        {
                            segment = new VoxelSegment(label);
                            currentSegmentLayer.Add(segment);
                        }

                        segment.Voxels.Add(World.GetVoxel(x, y, z));
                    }
                }

                SegmentLayers.Add(currentSegmentLayer);
            }
        }

        protected void PopulateElevators()
        {
            //Iterate segments top to bottom aka in reverse order
            for (int nSegmentLayer = SegmentLayers.Count - 1; nSegmentLayer >= 0; nSegmentLayer--)
            {
                var layer = SegmentLayers[nSegmentLayer];

                foreach (var segment in layer)
                {
                    var possibleElevators = new List<VoxelElevator>();

                    foreach (var voxel in segment.Voxels)
                    {
                        if (!World.TryGetVoxel(voxel.Position.X, voxel.Position.Y + 1, voxel.Position.Z, out var topVoxelTest) || topVoxelTest.Type == Voxel.VoxelType.Solid) continue; //From what ever direction the agent must be able to move into this voxel. Only wall can prevent that

                        var possiblePlacements = new List<Voxel>();
                        if (World.TryGetVoxel(voxel.Position.X - 1, voxel.Position.Y + 1, voxel.Position.Z, out var left)) possiblePlacements.Add(left);
                        if (World.TryGetVoxel(voxel.Position.X + 1, voxel.Position.Y + 1, voxel.Position.Z, out var right)) possiblePlacements.Add(right);
                        if (World.TryGetVoxel(voxel.Position.X, voxel.Position.Y + 1, voxel.Position.Z - 1, out var front)) possiblePlacements.Add(front);
                        if (World.TryGetVoxel(voxel.Position.X, voxel.Position.Y + 1, voxel.Position.Z + 1, out var back)) possiblePlacements.Add(back);

                        foreach (var checkPlacement in possiblePlacements)
                        {
                            if (checkPlacement.Type != Voxel.VoxelType.Empty) continue; //The possible placement block is not empty.

                            if (!World.TryGetVoxel(checkPlacement.Position.X, checkPlacement.Position.Y - 1, checkPlacement.Position.Z, out var topExitCheck) || topExitCheck.Type != Voxel.VoxelType.Empty) continue; //The block below the elevator exit on the top would not be empty. It makes no sense to place an elevator here

                            //Let down the rope
                            int possibleGoDown = 0;
                            while (true)
                            {
                                if (World.TryGetVoxel(checkPlacement.Position.X, checkPlacement.Position.Y - possibleGoDown - 1, checkPlacement.Position.Z, out var bttomVoxel))
                                {
                                    if (bttomVoxel.Type == Voxel.VoxelType.Empty)
                                    {
                                        possibleGoDown++;
                                        continue;
                                    }
                                }

                                break;
                            }

                            possibleElevators.Add(new VoxelElevator(checkPlacement, possibleGoDown));
                        }
                    }

                    var selectedElevators = new List<VoxelElevator>();

                    var WANTED_ELEVATORS = (int)Math.Ceiling(possibleElevators.Count * 0.1);

                    if (possibleElevators.Count > 0)
                    {
                        var observations = new double[possibleElevators.Count][];

                        for (int nElevator = 0; nElevator < possibleElevators.Count; nElevator++)
                        {
                            var end = possibleElevators[nElevator].ElevatorExitTop;
                            observations[nElevator] = new double[] { end.Position.X, end.Position.Y, end.Position.Z };
                        }

                        Accord.Math.Random.Generator.Seed = GenerationParameters.Seed;
                        var clusters = new KMeans(WANTED_ELEVATORS).Learn(observations);
                        var clusterLabels = clusters.Decide(observations);

                        foreach (var cluster in clusters)
                        {
                            var centerPos = new Vector3((float)cluster.Centroid[0], (float)cluster.Centroid[1], (float)cluster.Centroid[2]);

                            VoxelElevator closest = null;
                            Vector3 closestPos = default;
                            float closestDistanceSqr = default;

                            for (int nElevator = 0; nElevator < possibleElevators.Count; nElevator++)
                            {
                                if (clusterLabels[nElevator] != cluster.Index) continue;

                                var possibleElevator = possibleElevators[nElevator];

                                var possibleElevatorPos = new Vector3(possibleElevator.ElevatorExitTop.Position.X, possibleElevator.ElevatorExitTop.Position.Y, possibleElevator.ElevatorExitTop.Position.Z);

                                var dSqrPossibleToCenter = (possibleElevatorPos - closestPos).sqrMagnitude;

                                if (closest == null || dSqrPossibleToCenter < closestDistanceSqr)
                                {
                                    closest = possibleElevator;
                                    closestPos = possibleElevatorPos;
                                    closestDistanceSqr = dSqrPossibleToCenter;
                                }
                            }

                            if (closest != null)
                            {
                                selectedElevators.Add(closest);
                            }
                        }
                    }

                    foreach (var selectedElevator in selectedElevators)
                    {
                        for (int nStepDown = 0; nStepDown <= selectedElevator.StepsDown; nStepDown++)
                        {
                            int height = selectedElevator.ElevatorExitTop.Position.Y - nStepDown;

                            var voxel = World.GetVoxel(selectedElevator.ElevatorExitTop.Position.X, height, selectedElevator.ElevatorExitTop.Position.Z);

                            if (nStepDown == 0) //Top elevator exit aka 0 steps down
                            {
                                voxel.Type = Voxel.VoxelType.ElevatorShaft;
                            }
                            else if (nStepDown == selectedElevator.StepsDown) //Reached max step down. Spawn elevator entry
                            {
                                voxel.Type = Voxel.VoxelType.ElevatorRoot;
                                selectedElevator.ElevatorEntryBottom = voxel;
                                voxel.ElevatorHeight = nStepDown;
                            }
                            else
                            {
                                voxel.Type = Voxel.VoxelType.ElevatorShaft;
                            }
                        }
                    }
                }
            }
        }

        public void toRL()
        {
            rLLevel = new int[World.Width, World.Height - 1, World.Depth];

            for (int j = 0; j < World.Depth; j++)
            {
                for (int k = 1; k < World.Height; k++)
                {

                    for (int i = 0; i < World.Width; i++)
                    {

                        rLLevel[i, k - 1, j] = 11;

                        if (World.Voxels[i, k, j].Type == Voxel.VoxelType.Empty && World.Voxels[i, k - 1, j].Type == Voxel.VoxelType.Solid) rLLevel[i, k - 1, j] = 12;
                        if (World.Voxels[i, k, j].Type == Voxel.VoxelType.ElevatorRoot) rLLevel[i, k - 1, j] = 13;
                        if (World.Voxels[i, k, j].Type == Voxel.VoxelType.ElevatorShaft) rLLevel[i, k - 1, j] = 13;
                    }
                }
            }

            // rLLevel[(int)agentStart.x, (int)agentStart.y, (int)agentStart.z] = 14;
            //rLLevel[(int)agentGoal.x, (int)agentGoal.y, (int)agentGoal.z] = 14;
        }

        public Vector3Int Start;
        public Vector3Int Finish;
        public bool useGoal;
        public List<Node> shortestPath;
        public List<Node> AStarPath;
        public Node[,,] allNodes;
        protected Queue<Node> processingqueue;

        public double astarTime = 0;
        public double djikstraTime = 0;
        public Stopwatch timer = new Stopwatch();

        public bool aStar()
        {
            var astern = new Astern();
            astern.Dimensions = new Vector3Int(World.Width, World.Height, World.Depth);

            var result = astern.Findpath(rLLevel, Startpos, Endpos);
            astarTime = astern.timer.Elapsed.TotalMilliseconds;
            if (result.Count > 0) AStarPath = result;

            return result.Count != 0;
        }

        public void findDjikstra()
        {
            processingqueue = new Queue<Node>();
            allNodes = new Node[World.Width, World.Height - 1, World.Depth];
            for (int i = 0; i < World.Width; i++)
            {
                for (int k = 0; k < World.Height - 1; k++)
                {
                    for (int j = 0; j < World.Depth; j++)
                    {

                        allNodes[i, k, j] = new Node(new Vector3Int(i, k, j), int.MaxValue);
                        allNodes[i, k, j].Type = rLLevel[i, k, j];
                    }
                }
            }
            timer.Reset();
            timer.Start();
            Start = Startpos;
            allNodes[Start.X, Start.Y, Start.Z].Cost = 0;
            processingqueue.Enqueue(allNodes[Start.X, Start.Y, Start.Z]);

            bool finished = false;
            while (processingqueue.Any() && !finished)
            {
                findNeigbours(processingqueue.First());
                processingqueue.First().visited = true;
                if (processingqueue.First().Position.X == Finish.X && processingqueue.First().Position.Y == Finish.Y && processingqueue.First().Position.Z == Finish.Z)
                {
                    finished = true;
                    break;
                }

                // if (processingqueue.First().Source != null) processingqueue.First().Cost = processingqueue.First().Source.Cost + 1;
                allNodes[processingqueue.First().Position.X, processingqueue.First().Position.Y, processingqueue.First().Position.Z] = processingqueue.First();
                processingqueue.Dequeue();
            }

            Node furthestNode = allNodes[Start.X, Start.Y, Start.Z];

            if (TryGetNode(Finish.X, Finish.Y, Finish.Z, out Node goal)) { furthestNode = goal; } else Finish = furthestNode.Position;

            shortestPath = new List<Node>();
            Endpos = Finish;
            do
            {
                shortestPath.Add(furthestNode);
                if (furthestNode.Source != null)
                    furthestNode = furthestNode.Source;

            } while (furthestNode.Source != null);
            shortestPath.Add(new Node(Start, 0));
            shortestPath.Reverse();
            timer.Stop();
            djikstraTime = timer.Elapsed.TotalMilliseconds;
        }

        public void Findpath()
        {
            toRL();
            processingqueue = new Queue<Node>();
            allNodes = new Node[World.Width, World.Height - 1, World.Depth];
            for (int i = 0; i < World.Width; i++)
            {
                for (int k = 0; k < World.Height - 1; k++)
                {
                    for (int j = 0; j < World.Depth; j++)
                    {

                        allNodes[i, k, j] = new Node(new Vector3Int(i, k, j), int.MaxValue);
                        allNodes[i, k, j].Type = rLLevel[i, k, j];
                    }
                }
            }
            Start = Startpos;
            allNodes[Start.X, Start.Y, Start.Z].Cost = 0;
            processingqueue.Enqueue(allNodes[Start.X, Start.Y, Start.Z]);

            bool finished = false;
            while (processingqueue.Any() || finished)
            {
                findNeigbours(processingqueue.First());
                processingqueue.First().visited = true;
                //  if (processingqueue.First().Position.X == Finish.X && processingqueue.First().Position.Y == Finish.Y && processingqueue.First().Position.Z == Finish.Z && useGoal) continue;
                // if (processingqueue.First().Source != null) processingqueue.First().Cost = processingqueue.First().Source.Cost + 1;
                allNodes[processingqueue.First().Position.X, processingqueue.First().Position.Y, processingqueue.First().Position.Z] = processingqueue.First();
                processingqueue.Dequeue();
            }

            Node furthestNode = allNodes[Start.X, Start.Y, Start.Z];

            for (int i = 0; i < World.Width; i++)
            {
                for (int k = 0; k < World.Height - 1; k++)
                {
                    for (int j = 0; j < World.Depth; j++)
                    {

                        if (allNodes[i, k, j].Cost > furthestNode.Cost && allNodes[i, k, j].Cost != int.MaxValue)
                            furthestNode = allNodes[i, k, j];
                    }
                }
            }
            if (useGoal && TryGetNode(Finish.X, Finish.Y, Finish.Z, out Node goal)) { furthestNode = goal; } else Finish = furthestNode.Position;

            shortestPath = new List<Node>();
            Endpos = Finish;
            do
            {
                shortestPath.Add(furthestNode);
                if (furthestNode.Source != null)
                    furthestNode = furthestNode.Source;

            } while (furthestNode.Source != null);
            shortestPath.Add(new Node(Start, 0));
            shortestPath.Reverse();

            Console.WriteLine(shortestPath);

        }


        void findNeigbours(Node currentNode)
        {

            // if (TryGetNode(currentNode.Position.X - 1, currentNode.Position.Y, currentNode.Position.Z, out var left))  processingqueue.Enqueue(new Node(left.Position,currentNode.Cost + 1));
            // if (TryGetNode(currentNode.Position.X + 1, currentNode.Position.Y, currentNode.Position.Z, out var right)) processingqueue.Enqueue(new Node(right.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z - 1, out var front)) processingqueue.Enqueue(new Node(front.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z + 1, out var back)) processingqueue.Enqueue(new Node(back.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y -1, currentNode.Position.Z , out var bottom)) processingqueue.Enqueue(new Node(bottom.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y +1 , currentNode.Position.Z , out var top)) processingqueue.Enqueue(new Node(top.Position, currentNode.Cost + 1));

            if (TryGetNode(currentNode.Position.X - 1, currentNode.Position.Y, currentNode.Position.Z, out var left))
            {
                if (left.Type == 12 || left.Type == 13)
                    if (left.Cost > currentNode.Cost + 1)
                    {
                        left.Source = currentNode;
                        left.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(left);
                    };

            }
            if (TryGetNode(currentNode.Position.X + 1, currentNode.Position.Y, currentNode.Position.Z, out var right))
            {
                if (right.Type == 12 || right.Type == 13)
                    if (right.Cost > currentNode.Cost + 1)
                    {
                        right.Source = currentNode;
                        right.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(right);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z - 1, out var back))
            {
                if (back.Type == 12 || back.Type == 13)
                    if (back.Cost > currentNode.Cost + 1)
                    {
                        back.Source = currentNode;
                        back.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(back);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z + 1, out var front))
            {
                if (front.Type == 12 || front.Type == 13)
                    if (front.Cost > currentNode.Cost + 1)
                    {
                        front.Source = currentNode;
                        front.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(front);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y - 1, currentNode.Position.Z, out var bottom))
            {
                if (bottom.Type == 13)
                    if (bottom.Cost > currentNode.Cost + 1)
                    {
                        bottom.Source = currentNode;
                        bottom.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(bottom);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y + 1, currentNode.Position.Z, out var top))
            {
                if (top.Type == 13)
                    if (top.Cost > currentNode.Cost + 1)
                    {
                        top.Source = currentNode;
                        top.Cost = currentNode.Cost + 1;
                        processingqueue.Enqueue(top);
                    }

            }
            //Kosten wenn günstiger ersetzten
            //checken ob schon besucht

        }

        bool TryGetNode(Vector3Int Pos, out Node node)
        {
            node = default;

            if (Pos.X < 0 || Pos.X >= World.Width) return false;
            if (Pos.Y < 0 || Pos.Y >= World.Height - 1) return false;
            if (Pos.Z < 0 || Pos.Z >= World.Depth) return false;

            node = allNodes[Pos.X, Pos.Y, Pos.Z];
            return true;
        }
        bool TryGetNode(int x, int y, int z, out Node node)
        {
            node = default;

            if (x < 0 || x >= World.Width) return false;
            if (y < 0 || y >= World.Height - 1) return false;
            if (z < 0 || z >= World.Depth) return false;

            node = allNodes[x, y, z];
            return true;
        }
    }
}
