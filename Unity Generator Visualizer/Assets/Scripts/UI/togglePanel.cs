using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class togglePanel : MonoBehaviour
{
    // Start is called before the first frame update

    float startposition;
    float goalposition;
    public int moveamount;

    public bool hidden = false;
    bool moving = false;
    
    public float speed = 1.0f;
    void Start()
    {
        startposition = transform.position.x; 
        
    }

    // Update is called once per frame
    void Update()
    {
        if (moving)
        {
            transform.position = new Vector3(Mathf.Lerp(transform.position.x, goalposition,Time.deltaTime*speed), transform.position.y, transform.position.z);
            if(Math.Abs(goalposition - transform.position.x) < 10)
            {
                moving = false;
                transform.position = new Vector3(goalposition, transform.position.y, transform.position.z);
            }


        }
    }

    public void tooglePanel()
    {
        hidden = !hidden;
        moving = true;
        if (hidden) goalposition = startposition + moveamount;else
        goalposition =  hidden ? (startposition + moveamount) : startposition;
    }
}
