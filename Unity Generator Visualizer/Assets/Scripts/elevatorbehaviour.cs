using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class elevatorbehaviour : MonoBehaviour
{
    // Start is called before the first frame update
    public int maxHeight = 1;
    public PlayerManager Player;
    private Vector3 startposition;

    void Start()
    {
        Player = Object.FindObjectOfType<PlayerManager>();
        startposition = transform.position;
    }

    // Update is called once per frame
    void Update()
    {
       if(Player.height  <= startposition.y + maxHeight && Player.height >= startposition.y)
        transform.position = Vector3.Lerp(transform.position, new Vector3(startposition.x, Player.height, startposition.z),Time.deltaTime * 10f);
    }
}
