using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class pathlogik : MonoBehaviour
{
    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }
    //1 = backwards
    //2 = Center
    //3 = forward;
    //4 = left
    //5 = right
    //6 = up
    public void setPath(int before, int after)
    {
        
        Transform[] ts = gameObject.GetComponentsInChildren<Transform>();
        ts[0].localScale = Vector3.one *2;
        ts[1].localScale = Vector3.zero;
        //ts[2].localScale = Vector3.zero;
        ts[3].localScale = Vector3.zero;
        ts[4].localScale = Vector3.zero;
        ts[5].localScale = Vector3.zero;
        ts[6].localScale = Vector3.zero;
        if(before > 0)
        ts[before].localScale = Vector3.one*0.985f;
        if (after > 0)
            ts[after].localScale = Vector3.one * 0.985f;

    }
}
