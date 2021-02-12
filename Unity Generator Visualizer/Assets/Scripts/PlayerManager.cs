using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerManager : MonoBehaviour
{
    public GameObject player;
    public float speed = 1.0f;
    public float speed2 = 1.0f;
    public Vector3 offset;
    int queue = 0;
    public Vector3[] points;
    private Vector3 target;
    private Vector3 currentPos;
    private Vector3 nextposPos;
    public float height = 0;
    int upwards = 1;
    public LevelGenerator Level;
    public bool looping = false;
    float goalX, goalY;
    // 0 = idle, 1 = running
    public int State = 0;
    
    bool walking = false;

    public GameObject play;
    public GameObject pause;
    Animator playerAnim;
    // Start is called before the first frame update
    void Start()
    {
        playerAnim = player.GetComponent<Animator>();
    }

    // Update is called once per frame
    void Update()
    {
           
        currentPos = player.gameObject.transform.position;
        
        if (walking)
        {
            nextposPos = Vector3.MoveTowards(currentPos, target + offset, Time.deltaTime * speed) ;
            player.transform.position = nextposPos;
            var targetRotation = Quaternion.LookRotation(target + offset - player.transform.position);
            Debug.Log(target + offset - player.transform.position);
            float view = target.y + offset.y - player.transform.position.y;
            //if (Vector3.Dot(nextposPos.normalized, Vector3.up) < 0.7f && Vector3.Dot(nextposPos.normalized, Vector3.up) > -0.7f)
            if (view < 0.1f && view > -0.1f)
            {
                player.transform.rotation = Quaternion.Lerp(player.transform.rotation, targetRotation, speed2 * Time.deltaTime);
                playerAnim.SetInteger("state", 1);
            }
            else
            {
                playerAnim.SetInteger("state", 0);
            }
        }
        
        if(Vector3.Distance(currentPos, target + offset) < 0.05f)
        {
            player.transform.position = target + offset;
               queue = queue + upwards;
            if ((queue > points.Length - 1) || (queue == -1))
            {
                
                    upwards = upwards * -1;
                
                    playerAnim.SetInteger("state", 0);

                if (!looping)
                {
                    playerAnim.SetInteger("state", 0);
                    walking = false;
                    pause.SetActive(false);
                    play.SetActive(true);
                }
            }
            if(queue < points.Length && queue > -1)
            target = points[queue];
            Debug.Log(queue);
        }

        height = player.transform.position.y;
    }
    public void run()
    {
        loadcoordinats();
        player.gameObject.transform.position = points[queue] + offset;
        playerAnim.SetInteger("state", 1);
        target = points[queue];
        walking = true;
    }

    public void stop()
    {
        walking = false;
        playerAnim.SetInteger("state", 0);
    }

    public void reset()
    {
        player.gameObject.transform.position = points[0] + offset;
        target = points[1] + offset;
        walking = false;
        playerAnim.SetInteger("state", 0);
        queue = 0;
        upwards = 1;

    }

    public void toogleLoop()
    {
        looping = !looping;
    }

    public void arrive()
    {
        playerAnim.SetInteger("state", 0);
        target = points[queue];
        walking = false;
    }

    public void elevator()
    {
        playerAnim.SetInteger("state", 0);
        target = points[queue];
        walking = false;
    }

    public void loadcoordinats()
    {
        Level.GenerateVoxelWorld();
        points = new Vector3[Level.DjikstraPath.Count];
        for(int i = 0; i< Level.DjikstraPath.Count; i++)
        {
            points[i] = Level.DjikstraPath[i];
        }
    }

    
}
