using System.Collections;
using System.Collections.Generic;
using TMPro;
using UnityEngine;
using UnityEngine.UI;
using static LevelGenerator;

public class UIManager : MonoBehaviour
{
    public LevelGenerator levelgenerator;
    public Pathmanager paths;
    List<LevelData> Leveldata;
    //// Create
    Vector3 dimensions;
    Vector3Int startPosition;
    Vector3Int endPosition;
    public bool random;
    public bool automatic;

    public int Typechoosen;


    /// Info
    public TextMeshProUGUI Seed;
    public TextMeshProUGUI Dimension;
    public TextMeshProUGUI Gantime;
    public TextMeshProUGUI RLtime;
    public TextMeshProUGUI Asterntime;
    public TextMeshProUGUI Djikstratime;
    public TextMeshProUGUI TFAtime;
    public TextMeshProUGUI GanSteps;
    public TextMeshProUGUI RLSteps;
    public TextMeshProUGUI AsternSteps;
    public TextMeshProUGUI DjikstraSteps;
    public TextMeshProUGUI TFASteps;

    public bool showGan;
    public bool showRL;
    public bool showAstern;

    public int usePath;

    /// Load
    public TextMeshProUGUI SeedL;
    public TextMeshProUGUI DimensionL;
    public TextMeshProUGUI LevelTypeL;
    public TextMeshProUGUI StartposL;
    public TextMeshProUGUI EndposL;
    public TMP_Dropdown Levelloader;

    
    public int loadLevel = 0;

    int seed;
    // Start is called before the first frame update
    void Start()
    {
        dimensions = new Vector3();
        startPosition = new Vector3Int();
        endPosition = new Vector3Int();
    }

    // Update is called once per frame
    void Update()
    {

    }

    public void createLevel()
    {
        levelgenerator.seed = seed;
        levelgenerator.width = (int)dimensions.x;
        levelgenerator.height = (int)dimensions.y;
        levelgenerator.depth = (int)dimensions.z;
        levelgenerator.agentStart = startPosition;
        levelgenerator.agentGoal = endPosition;
        levelgenerator.useGoal = !automatic;
        levelgenerator.randomStart = random;
        levelgenerator.leveltype = (LevelGenerator.LevelType)Typechoosen;
        levelgenerator.GenerateVoxelWorld(); 
    }

    public void LoadLevel()
    {

    }

    public void saveLevel()
    {
        levelgenerator.ExportLevel();
    }

    public void loadLevelList(List<LevelData> Levels)
    {
        Levelloader.ClearOptions();
        Leveldata = new List<LevelData>();
        List<string> names = new List<string>();
        foreach(var level in Levels)
        {
            names.Add(level.name);
        }
        Leveldata = Levels;
        Levelloader.AddOptions(names);
    }

    public void setLeveltoLoad()
    {

    }

    #region setter
    public void setDimensionsX(string x)
    {
        dimensions.x = int.Parse(x);
    }
    public void setDimensionsY(string y)
    {
        dimensions.y = int.Parse(y);
    }
    public void setDimensionsZ(string z)
    {
        dimensions.z = int.Parse(z);
    }

    public void setStartX(string x)
    {
        startPosition.z = int.Parse(x);
    }
    public void setStartY(string y)
    {
        startPosition.y = int.Parse(y);
    }
    public void setStartZ(string z)
    {
        startPosition.x = int.Parse(z);
    }

    public void setEndX(string x)
    {
        endPosition.z = int.Parse(x);
    }
    public void setEndY(string y)
    {
        endPosition.y = int.Parse(y);
    }
    public void setEndZ(string z)
    {
        endPosition.x = int.Parse(z);
    }


    public void setSeed(string _seed)
    {
        seed = int.Parse(_seed);
    }

    public void automate(bool _automatic){
        automatic = _automatic;
        levelgenerator.useGoal = !_automatic;
    }
    public void randomer(bool _random)
    {
        random = _random;
        levelgenerator.randomStart = _random;
    }

    public void Gan(bool _gan)
    {
        showGan = _gan;
        paths.Paths[0].SetActive(_gan);
    }

    public void RL(bool _rl)
    {
        showRL = _rl;
        paths.Paths[1].SetActive(_rl);
    }

    public void TL(bool _tl)
    {
        
        paths.Paths[2].SetActive(_tl);
    }

    public void DJ(bool _djk)
    {
        
        paths.Paths[3].SetActive(_djk);
    }
    public void AS(bool _as)
    {
        showAstern = _as;
        paths.Paths[4].SetActive(_as);
    }

    public void SetPath(int x)
    {
        usePath = x;
    }

    public void chooseLevel(int x)
    {
        loadLevel = x;
        
        SeedL.text = Leveldata[x].levelseed.ToString();
        DimensionL.text = Leveldata[x].xdimensions[0] + "x" +  Leveldata[x].xdimensions[1] + "x" + Leveldata[x].xdimensions[2];
        LevelTypeL.text = Leveldata[x].type;
        StartposL.text = Leveldata[x].agentStart[0] + "x" + Leveldata[x].agentStart[1] + "x" + Leveldata[x].agentStart[2];
        EndposL.text = Leveldata[x].agentGoal[0] + "x" + Leveldata[x].agentGoal[1] + "x" + Leveldata[x].agentGoal[2];

    }

    public void fillInfo()
    {
        Seed.text = levelgenerator.seed.ToString();
        Dimension.text = levelgenerator.width + "x" + levelgenerator.height + "x" + levelgenerator.depth;
        Gantime.text = levelgenerator.GanTime.ToString("0.00") + "ms";
        RLtime.text = levelgenerator.RLTime.ToString("0.00") + "ms";
        Asterntime.text = levelgenerator.AStarTime.ToString("0.00") + "ms";
        Djikstratime.text = levelgenerator.djikstraTime.ToString("0.00") + "ms";
        TFAtime.text = levelgenerator.TLTime.ToString("0.00") + "ms";
        GanSteps.text = levelgenerator.Gansteps.ToString() + " steps";
        RLSteps.text = levelgenerator.RLSteps.ToString() + " steps";
        AsternSteps.text = levelgenerator.aStarSteps.ToString() + " steps";
        DjikstraSteps.text = levelgenerator.djikstraSteps.ToString() + " steps";
        TFASteps.text = levelgenerator.TLSteps.ToString() + " steps";
    }

    public void importLevel()
    {
        levelgenerator.loadLevel(loadLevel); 
    }

    public void chooseLevelType(int x)
    {
        Typechoosen = x;
    }
    #endregion
}
