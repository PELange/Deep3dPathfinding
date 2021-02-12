using Assets.Scripts.VoxelWorld;
using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using TMPro;
using System.Diagnostics;

public class LevelGenerator : MonoBehaviour
{
    public TextAsset Levelpath;
    //wall Bridge Stair empty
    public Transform Parent;
    public GameObject[] VoxelObjects;

    public GameObject[] Archetypes;
    public GameObject[] DebugVoxels;
    public GameObject[] Startfinish;
    public GameObject Player;
    public UIManager UI;



    // 0-3 , 0 = north (z+) east (x+) south (z-) west (x-)   Direction of the Stairs in relation to the axis. z+ points in the direction of positiv z, aka north
    public GameObject[] Stairs;
    public LevelType leveltype;
    public bool randomType = false;
    GameObject[] Layers;
    public int width = 5;
    public int height = 1;
    public int depth = 5;


    private int _width;
    private int _height;
    private int _depth;

    public int seed = 0;
    public int pathsperlevel = 5;
    public bool walls = true;

    public bool randomStart = false;
    public UnityEngine.Vector3Int agentStart;
    public bool useGoal = false;
    public UnityEngine.Vector3Int agentGoal;
    public List<Vector3> RLPath;
    public List<Vector3> DjikstraPath;
    public List<Vector3> AstarPath;
    public List<Vector3> GanPath;

    public int Levelnumber = 10;

    private Pathmanager pathviz;

    private List<LevelExport> GANList;
    private List<LevelExport> RLList;
    private List<LevelExport> DJKList;
    private List<LevelExport> TLList;
    private List<LevelData> InterfaceList;
    public List<String> Levelnames;

    /// measureTime, turn off if not using
    public bool measuringTime = true;
    public double djikstraTime = 0;
    public int djikstraSteps = 0;
    public double AStarTime = 0;
    public int aStarSteps = 0;
    public double GanTime = 0;
    public int Gansteps = 0;
    public double RLTime = 0;
    public int RLSteps = 0;
    public double TLTime = 0;
    public int TLSteps = 0;

    // 0 = wall , 1 = empty, 2 = Bridge, 10-13 Stairs, 14 = Elevatorroot, 15 = elevatorshaft
    public enum VoxelType
    {
        None = -1,
        Wall = 0,
        Empty = 1,
        Bridge = 2,
        ElevatorRoot = 14,
        ElevatorShaft = 15,
    }
    int[,,] level;

    [System.Serializable]
    public enum LevelType
    {
        Noise = 0,
        Canyon = 1,
        Maze = 2,
        Hills = 3,
        smallHills = 4
    }
    // 9 = Resultpathvoxel
    // 10 = currentagentvoxel
    // 11 = Barrier
    // 12 = Empty
    // 13 = Elevator
    // 14 = Start/EndVoxel
    int[,,] rLLevel;



    
    void Start()
    {
        GANList = new List<LevelExport>();
        RLList = new List<LevelExport>();
        DJKList = new List<LevelExport>();
        TLList = new List<LevelExport>();
        InterfaceList = new List<LevelData>();
        
        pathviz = GetComponent<Pathmanager>();
        level = new int[_width, _height, _depth];
        rLLevel = new int[_width, height, _depth];
        importLevels();

        UI.loadLevelList(InterfaceList);
        GenerateVoxelWorld();
    }

    public void getdimensions()
    {
        _width = width;
        _height = height + 1;
        _depth = depth;
    }
    public void GenerateVoxelWorld()
    {
        getdimensions();
        pathviz.clearpaths();
        //Clear old level
        foreach (Transform child in Parent)
        {
            Destroy(child.gameObject);
        }

        //Setup new layers
        Layers = new GameObject[_height];
        for (int i = 0; i < _height; i++)
        {
            var layer = new GameObject
            {
                name = i.ToString()
            };
            layer.transform.SetParent(Parent);
            Layers[i] = layer;
        }
        
        //Generate world
        var generator = VoxelWorldGenerator.Run(new VoxelWorldGenerator.Parameters
        {
            Width = _width,
            Height = _height,
            Depth = _depth,
            Seed = seed,
            levelType = (VoxelWorldGenerator.LevelType)leveltype
        });
        generator.toRL();


        generator.Startpos = new Assets.Scripts.VoxelWorld.Vector3Int(agentStart.x, agentStart.y, agentStart.z);
        
        if (useGoal)
        {
            generator.Finish = new Assets.Scripts.VoxelWorld.Vector3Int(agentGoal.x, agentGoal.y, agentGoal.z);
            generator.useGoal = true;
        }

        var formerstarts = new List<UnityEngine.Vector3Int>();
        if (randomStart)
        {   int abboard = 0;
            bool positionfound = false;
            var start = new UnityEngine.Vector3Int(UnityEngine.Random.Range(0, width), UnityEngine.Random.Range(0, height), UnityEngine.Random.Range(0, depth));
            while (!positionfound && abboard < 15)
            {

                if (!(generator.rLLevel[start.x, start.y, start.z] == 11) && !formerstarts.Contains(start))
                {
                    positionfound = true;
                    break;
                }
                start = new UnityEngine.Vector3Int((int)UnityEngine.Random.Range(0, width), (int)UnityEngine.Random.Range(0, height), (int)UnityEngine.Random.Range(0, depth));
                abboard++;
                formerstarts.Add(start);
            }
            
            generator.Startpos = new Assets.Scripts.VoxelWorld.Vector3Int(start.x, start.y, start.z);
            agentStart = new UnityEngine.Vector3Int(start.x, start.y, start.z);
        }

        generator.Findpath();
       // 
        if (!useGoal) { 
            agentGoal = new UnityEngine.Vector3Int(generator.Finish.X, generator.Finish.Y, generator.Finish.Z);
        }
        rLLevel = generator.rLLevel;


        // generates mean of 1000 scan, turn off if not using
        if (measuringTime)
        {
            double astarSum = 0;
            double djikstraSum = 0;
            for (int t = 0; t < 1000; t++)
            {
                generator.aStar();
                astarSum += generator.astarTime;
                generator.findDjikstra();
                djikstraSum += generator.djikstraTime;
            }
            AStarTime = astarSum / 1000;
            aStarSteps = generator.AStarPath.Count;
            djikstraTime = djikstraSum / 1000;
            djikstraSteps = generator.shortestPath.Count;
        }
        else
        {
            generator.aStar();
            AStarTime = generator.astarTime;
            aStarSteps = generator.AStarPath.Count;
            generator.findDjikstra();
            djikstraTime = generator.djikstraTime;
            djikstraSteps = generator.shortestPath.Count;
        }

        //Render world
        foreach (var voxel in generator.World.Voxels)
        {
            //Do not render empty blocks or elevator shafts.
            if (voxel.Type == Voxel.VoxelType.Empty || voxel.Type == Voxel.VoxelType.ElevatorShaft) continue;

            var obj = Instantiate(VoxelObjects[(int)voxel.Type], new Vector3(voxel.Position.Z, voxel.Position.Y, voxel.Position.X), Quaternion.identity, Layers[voxel.Position.Y].transform);

            if (voxel.Type == Voxel.VoxelType.ElevatorRoot)
            {

                obj.transform.GetComponent<elevatorbehaviour>().maxHeight = voxel.ElevatorHeight;
            }
            
        }
        DjikstraPath.Clear();
        foreach (Node pos in generator.shortestPath)
        {
            DjikstraPath.Add(new Vector3(pos.Position.Z, pos.Position.Y + 1, pos.Position.X));
        }
        pathviz.drawPath(4, DjikstraPath);
        AstarPath.Clear();
        foreach (Node pos in generator.AStarPath)
        {
            AstarPath.Add(new Vector3(pos.Position.Z, pos.Position.Y + 1, pos.Position.X));
        }
        pathviz.drawPath(3, AstarPath);
        setstartfinish();
        
        
    }

    public void ExportLevel()
    {
        int[] transferlevel = new int[_width * (_height - 1) * _depth];
        int[] transferPath = new int[DjikstraPath.Count * 3];
        int j = 0;
        for (int c = 0; c < _height - 1; c++)
        {
            for (int a = 0; a < _width; a++)
            {
                for (int b = 0; b < _depth; b++)
                {


                    transferlevel[j] = rLLevel[a, c, b];
                    j++;

                }
            }
        }

        int k = 0;
        foreach (Vector3 pos in DjikstraPath)
        {
            transferPath[k] = (int)pos.x;
            transferPath[k + 1] = (int)pos.y - 1;
            transferPath[k + 2] = (int)pos.z;
            k += 3;
        }
        LevelExport levelExport = new LevelExport
        {
            xdimensions = new int[] { _depth, height, _width },
            levelseed = seed,
            exportlevel = transferlevel,
            DjikstraTime = djikstraTime,
            aStarTime = AStarTime,
            djikstraPath = transferPath,
            type = leveltype.ToString(),
            agentStart = new int[] { (int)agentStart.z, (int)agentStart.y, (int)agentStart.x },
            agentGoal = new int[] { (int)agentGoal.z, (int)agentGoal.y, (int)agentGoal.x }
        };

        string json = JsonUtility.ToJson(levelExport, true);

        var basePath = Application.dataPath + "/Jsons" + "/Generator/";

        Directory.CreateDirectory(basePath);

        File.WriteAllText(basePath + _depth + "x" + (_height - 1) + "x" + _width + "_" + seed + "_paths_" + ".json", json);
        
        importLevels();
        UI.loadLevelList(InterfaceList);

    }

    public void showLayer(float amount)
    {
        for (int i = 0; i < _height; i++)
        {
            if (amount < (float)i / _height) Layers[i].active = false; else Layers[i].active = true;
        }
    }

    public void setstartfinish()
    {
        var obj = Instantiate(Archetypes[5], new Vector3(agentStart.z, agentStart.y+1, agentStart.x), Quaternion.identity, Layers[agentStart.y].transform);
        var obj1 = Instantiate(Archetypes[6], new Vector3(agentGoal.z, agentGoal.y+1, agentGoal.x), Quaternion.identity, Layers[agentGoal.y].transform);
        Player.transform.position = new Vector3(agentStart.z + 0.5f, agentStart.y + 1, agentStart.x +0.5f);

    }

    public bool testLevel()
    {
        return true;
    }

    public void generateLevels()
    {
        
        getdimensions();

        var failedLevels = 0;
        for (int i = seed; i < seed + Levelnumber + failedLevels; i++)
        {
            var selectedType = randomType ? (LevelType)UnityEngine.Random.Range(3, 5) : leveltype;
            
            int abboard = 0;

            var formerstarts = new List<UnityEngine.Vector3Int>();
            var start = new UnityEngine.Vector3Int(UnityEngine.Random.Range(0, width), UnityEngine.Random.Range(0, height), UnityEngine.Random.Range(0, depth));

            var generator = VoxelWorldGenerator.Run(new VoxelWorldGenerator.Parameters
            {
                Width = _width,
                Height = _height,
                Depth = _depth,
                levelType = (VoxelWorldGenerator.LevelType)selectedType,
                Seed = i
            }); ;

            generator.toRL();
            var failedPaths = 0;
            var noPathsInLevelPossible = false;

            for (int nPathAttempt = 0; nPathAttempt < pathsperlevel + failedPaths; nPathAttempt++)
            {
                if (failedPaths >= pathsperlevel)
                {
                    noPathsInLevelPossible = true;
                    break;
                }

                bool positionfound = false;
                while (!positionfound && abboard < 15)
                {

                    if (!(generator.rLLevel[start.x, start.y, start.z] == 11) && !formerstarts.Contains(start))
                    {
                        positionfound = true;
                        break;
                    }
                    start = new UnityEngine.Vector3Int((int)UnityEngine.Random.Range(0, width), (int)UnityEngine.Random.Range(0, height), (int)UnityEngine.Random.Range(0, depth));
                    abboard++;
                }
                formerstarts.Add(start);
                generator.Startpos = new Assets.Scripts.VoxelWorld.Vector3Int(start.x, start.y, start.z);

                generator.Findpath();
                

                if (generator.shortestPath.Count < 2 || (generator.Start.X == generator.Finish.X && generator.Start.Y == generator.Finish.Y && generator.Start.Z == generator.Finish.Z))
                {
                    failedPaths++;
                    continue;
                }

                var finish = new UnityEngine.Vector3Int(generator.Finish.X, generator.Finish.Y, generator.Finish.Z);

                List<Vector3> solutionpath = new List<Vector3>();
                foreach (Node pos in generator.shortestPath)
                {
                    solutionpath.Add(new Vector3(pos.Position.Z, pos.Position.Y + 1, pos.Position.X));
                }

                
                int[] transferlevel = new int[_width * (_height - 1) * _depth];
                int[] transferPath = new int[solutionpath.Count * 3];
                int j = 0;
                for (int c = 0; c < _height - 1; c++)
                {
                    for (int a = 0; a < _width; a++)
                    {
                        for (int b = 0; b < _depth; b++)
                        {


                            transferlevel[j] = generator.rLLevel[a, c, b];
                            j++;

                        }
                    }
                }

                int k = 0;
                foreach (Vector3 pos in solutionpath)
                {
                    transferPath[k] = (int)pos.x;
                    transferPath[k + 1] = (int)pos.y - 1;
                    transferPath[k + 2] = (int)pos.z;
                    k += 3;
                }

                

                LevelExport levelExport = new LevelExport
                {
                    xdimensions = new int[] { _depth, height, _width },
                    levelseed = i,
                    exportlevel = transferlevel,
                    djikstraPath = transferPath,
                    type = selectedType.ToString(),
                    blockedPercentage = generator.blockedPercentage,
                    agentStart = new int[] { (int)start.z, (int)start.y, (int)start.x },
                    agentGoal = new int[] { (int)finish.z, (int)finish.y, (int)finish.x }
                };

                string json = JsonUtility.ToJson(levelExport, true);

                var basePath = Application.dataPath + "/Jsons" + "/";

                Directory.CreateDirectory(basePath);

                File.WriteAllText(basePath + _depth + "x" + (_height - 1) + "x" + _width + "_" + i + "_paths_" + nPathAttempt + ".json", json);
            }

            if (noPathsInLevelPossible)
            {
                failedLevels++;
            }
        }
    }

    [Serializable]
    private class LevelExport
    {
        public int[] xdimensions;
        public int levelseed;
        public string type;
        public int blockedPercentage;
        public int[] agentStart;
        public int[] agentGoal;
        public double DjikstraTime;
        public int[] djikstraPath;
        public double aStarTime;
        public int[] aStarpath;
        public int[] exportlevel;
        public List<int> resultPath;
        public List<int> resultVoxel;
        public double resultTime;
    }

    public class LevelData
    {
        public string name;
        public int[] xdimensions;
        public int levelseed;
        public string type;
        public int blockedPercentage;
        public int[] agentStart;
        public int[] agentGoal;
        public int[] djikstraPath;
        public int[] exportlevel;
        public List<int> resultPath;
        public List<int> resultVoxel;
    }


    [Serializable]
    private class PathImport
    {
        public List<int> resultPath;
    }

    [Serializable]
    public class coords
    {
        //  public List<int>;
    }

    public List<Vector3> myimportPath;
    public void importPath()
    {
        string jsontext = Levelpath.text;
        LevelExport importPath = JsonUtility.FromJson<LevelExport>(jsontext);
       
        GanPath.Clear();
        int i = 0;
        //while (i < importPath.resultPath.Count)
        //{
        //    Vector3 pos = new Vector3(importPath.resultPath[i], importPath.resultPath[i + 1] + 1, importPath.resultPath[i + 2]);
        //    GanPath.Add(pos);
        //    i = i + 3;
        //}

        while (i < importPath.resultVoxel.Count)
        {
            Vector3 pos = new Vector3(importPath.resultVoxel[i], importPath.resultVoxel[i + 1] + 1, importPath.resultVoxel[i + 2]);
            GanPath.Add(pos);
            i = i + 3;
        }
        pathviz.drawPath(1, GanPath);
    }

    public void findmatchingPaths(int Type)
    {
        
        switch (Type)
        {
            case 4:
                var result = DJKList.Find(x => x.levelseed == seed && x.xdimensions[0] == width && x.xdimensions[1] == height && x.xdimensions[2] == depth);
                if (result == null) break;

                int i = 0;
                //if (!measuringTime) { 
                //djikstraTime = result.DjikstraTime;
                //djikstraSteps = result.djikstraPath.Length/3;
                //AStarTime = result.aStarTime;
                //aStarSteps = result.aStarpath.Length/3;
                //}

                List<Vector3> DJPath = new List<Vector3>();
                while (i < result.djikstraPath.Length)
                {
                    Vector3 pos = new Vector3(result.djikstraPath[i], result.djikstraPath[i + 1] + 1, result.djikstraPath[i + 2]);
                    DJPath.Add(pos);
                    i = i + 3;
                }
                // pathviz.drawPath(3, DJPath);

                List<Vector3> ASPath = new List<Vector3>();
                while (i < result.aStarpath.Length)
                {
                    Vector3 pos = new Vector3(result.aStarpath[i], result.aStarpath[i + 1] + 1, result.aStarpath[i + 2]);
                    ASPath.Add(pos);
                    i = i + 3;
                }
               // pathviz.drawPath(4, ASPath);
                break;
            case 0:
                var GANresult = GANList.Find(x => x.levelseed == seed && x.xdimensions[0] == width && x.xdimensions[1] == height && x.xdimensions[2] == depth);
                if (GANresult == null)
                {
                    GanTime = 0;
                    Gansteps = 0;
                    break;
                }

                GanTime = GANresult.resultTime;
                Gansteps = GANresult.resultPath.Count/3;
                int k = 0;
                List<Vector3> GanPath = new List<Vector3>();
                while (k < GANresult.resultPath.Count)
                {
                    Vector3 pos = new Vector3(GANresult.resultPath[k], GANresult.resultPath[k + 1] + 1, GANresult.resultPath[k + 2]);
                    GanPath.Add(pos);
                    k = k + 3;
                }
                pathviz.drawPath(0, GanPath); break;
            case 1:
                var rlresult = RLList.Find(x => x.levelseed == seed && x.xdimensions[0] == width && x.xdimensions[1] == height && x.xdimensions[2] == depth);
                if (rlresult == null) {
                    RLTime = 0;
                    RLSteps = 0;
                    break;
                }
                RLTime = rlresult.resultTime;
                RLSteps = rlresult.resultPath.Count/3;
                int j = 0;
                List<Vector3> RLPath = new List<Vector3>();
                while (j < rlresult.resultPath.Count)
                {
                    Vector3 pos = new Vector3(rlresult.resultPath[j], rlresult.resultPath[j + 1] + 1, rlresult.resultPath[j + 2]);
                    RLPath.Add(pos);
                    j = j + 3;
                }
                pathviz.drawPath(1, RLPath);
                break;
            case 2:
                var tfaresult = TLList.Find(x => x.levelseed == seed && x.xdimensions[0] == width && x.xdimensions[1] == height && x.xdimensions[2] == depth);
                if (tfaresult == null)
                {
                    TLTime = 0;
                    TLSteps = 0;
                    break;
                }
                TLTime = tfaresult.resultTime;
                TLSteps = tfaresult.resultPath.Count/3;
                int l = 0;
                List<Vector3> tfaPath = new List<Vector3>();
                while (l < tfaresult.resultPath.Count)
                {
                    Vector3 pos = new Vector3(tfaresult.resultPath[l], tfaresult.resultPath[l + 1] + 1, tfaresult.resultPath[l + 2]);
                    tfaPath.Add(pos);
                    l = l + 3;
                }
                pathviz.drawPath(2, tfaPath);
                break;

        }
        UI.fillInfo();

    }


    public void importLevels()
    {
        DJKList.Clear();
        GANList.Clear();
        RLList.Clear();
        DJKList.Clear();
        TLList.Clear();

        var ganPath = Application.dataPath + "/Jsons" + "/GAN/";
        foreach (var file in Directory.EnumerateFiles(ganPath, "*.json"))
        {
            string dataAsJson = File.ReadAllText(file);
            LevelExport importPath = JsonUtility.FromJson<LevelExport>(dataAsJson);
            GANList.Add(importPath);
        }
        var RLPath = Application.dataPath + "/Jsons" + "/RL/";
        foreach (var file in Directory.EnumerateFiles(RLPath, "*.json"))
        {
            string dataAsJson = File.ReadAllText(file);
            LevelExport importPath = JsonUtility.FromJson<LevelExport>(dataAsJson);
            RLList.Add(importPath);
        }
        var TLPath = Application.dataPath + "/Jsons" + "/TL/";
        foreach (var file in Directory.EnumerateFiles(TLPath, "*.json"))
        {
            string dataAsJson = File.ReadAllText(file);
            LevelExport importPath = JsonUtility.FromJson<LevelExport>(dataAsJson);
            TLList.Add(importPath);
        }
        var DJPath = Application.dataPath + "/Jsons" + "/Generator/";
        foreach (var file in Directory.EnumerateFiles(DJPath, "*.json"))
        {
            string dataAsJson = File.ReadAllText(file);
            LevelExport importPath = JsonUtility.FromJson<LevelExport>(dataAsJson);
            LevelData data = new LevelData();
            data.xdimensions = importPath.xdimensions;
            data.levelseed = importPath.levelseed;
            data.agentStart = importPath.agentStart;
            data.agentGoal = importPath.agentGoal;
            data.type = importPath.type;
            data.name = Path.GetFileName(file);
            InterfaceList.Add(data);
            DJKList.Add(importPath);
        }

        
    }

    public void loadLevel(int x)
    {
        seed = DJKList[x].levelseed;
        width = DJKList[x].xdimensions[0];
        height = DJKList[x].xdimensions[1];
        depth = DJKList[x].xdimensions[2];
        agentStart = new UnityEngine.Vector3Int(DJKList[x].agentStart[2], DJKList[x].agentStart[1],DJKList[x].agentStart[0]);
        agentGoal = new UnityEngine.Vector3Int(DJKList[x].agentGoal[2], DJKList[x].agentGoal[1], DJKList[x].agentGoal[0]);
        useGoal = true;
        randomStart = false;
        leveltype = (LevelType)Enum.Parse(typeof(LevelType), DJKList[x].type, true);
        GenerateVoxelWorld();
        pathviz.clearpaths();
        findmatchingPaths(0);
        findmatchingPaths(1);
        findmatchingPaths(2);
        pathviz.drawPath(3, DjikstraPath);
        pathviz.drawPath(4, AstarPath);

    }
}