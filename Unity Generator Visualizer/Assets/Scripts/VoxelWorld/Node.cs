using UnityEngine;
using UnityEditor;

namespace Assets.Scripts.VoxelWorld
{
    public class Node
    {
        public Assets.Scripts.VoxelWorld.Vector3Int Position { get; set; }
        public int Cost = 0;
        public Node Source { get; set; }
        public int Index;
        public int Type;
        public float Heuristic;

        public bool visited { get; set; }
        public Node(Assets.Scripts.VoxelWorld.Vector3Int position, int cost)
        {
            Position = position;
            Cost = cost;
        }

        public Node(int x, int y, int height, int cost)
        {
            Position = new Assets.Scripts.VoxelWorld.Vector3Int(x, y, height);
            Cost = cost;
        }

        public Node(int x, int y, int height, int cost, float heuristic)
        {
            Position = new Assets.Scripts.VoxelWorld.Vector3Int(x, y, height);
            Cost = cost;
            Heuristic = heuristic;
        }


    }
}