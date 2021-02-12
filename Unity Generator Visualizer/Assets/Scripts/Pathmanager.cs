using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class Pathmanager : MonoBehaviour
{

    public List<Vector3> RLPath;
    public List<Vector3> Astern;
    public List<Vector3> Ganpath;
    public List<Vector3> TFAPath;

    public GameObject[] Pathnode;
    
    public GameObject[] Paths;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void clearpaths()
    {
        foreach (Transform child in Paths[0].transform)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in Paths[1].transform)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in Paths[2].transform)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in Paths[3].transform)
        {
            Destroy(child.gameObject);
        }
        foreach (Transform child in Paths[4].transform)
        {
            Destroy(child.gameObject);
        }
    }

    public void drawPath(int type, List<Vector3>Path)
    {
        
        Vector3 before = Vector3.zero;
        Vector3 after = Vector3.zero;
        for (int i = 0; i < Path.Count; i++)
        {
            int diff1 = -1, diff2 = -1;
            if (i > 0)
            {
                before = Path[i] - Path[i - 1];
                if (before.x == -1f) diff1 = 4;
                if (before.x == 1f) diff1 = 5;
                if (before.z == -1f) diff1 = 1;
                if (before.z == 1f) diff1 = 3;
                if (before.y == -1f) diff1 = 6;
                Debug.Log(before);
            }
            if (i < Path.Count - 1)
            {
                after = Path[i + 1] - Path[i];
                if (after.x == 1f) diff2 = 4;
                if (after.x == -1f) diff2 = 5;
                if (after.z == 1f) diff2 = 1;
                if (after.z == -1f) diff2 = 3;
                if (after.y == 1f) diff2 = 6;
            }
            var obj = Instantiate(Pathnode[type], new Vector3(Path[i].x + 0.5f, Path[i].y, Path[i].z + 0.5f), Quaternion.Euler(-90, 0, 0), Paths[type].transform);
            obj.GetComponent<pathlogik>().setPath(diff1, diff2);
        }
       
    }
}
