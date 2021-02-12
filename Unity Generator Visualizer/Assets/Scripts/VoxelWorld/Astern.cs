using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using UnityEngine;

namespace Assets.Scripts.VoxelWorld
{
    class Astern
    {
        public Vector3Int Dimensions;
        public Vector3Int Start;
        public Vector3Int Finish;
        Vector3 Goal;
        
        public List<Node> shortestPath;
        public Node[,,] allNodes;
        protected List<Node> openList;
        protected List<Node> closedList;
        int[,,] rLLevel;
        int Width, Height, Depth;

        public List<Node> OpenSet { get; set; } = new List<Node>();
        public List<Node> ClosedSet { get; set; } = new List<Node>();
        public Dictionary<Node, Node> CameFrom { get; set; } = new Dictionary<Node, Node>();
        public Dictionary<Node, double> GScore { get; set; } = new Dictionary<Node, double>();
        public Dictionary<Node, double> FScore { get; set; } = new Dictionary<Node, double>();
        public Stopwatch timer = new Stopwatch();

        public List<Node> BackTrack(Node current)
        {
            var resultPath = new List<Node>
            {
                current
            };

            while (CameFrom.ContainsKey(current))
            {
                current = CameFrom[current];
                resultPath.Insert(0, current);
            }
            timer.Stop();
            return resultPath;
        }

        public List<Node> Findpath(int[,,] _RLLlevel,Vector3Int Startnode, Vector3Int Endnode)
        {
            Goal = new Vector3(Endnode.X, Endnode.Y, Endnode.Z);
            Start = Startnode;
            Finish = Endnode;
            Width = Dimensions.X;
            Height = Dimensions.Y;
            Depth = Dimensions.Z;

            openList = new List<Node>( );
            allNodes = new Node[Dimensions.X, Dimensions.Y - 1, Dimensions.Z];
            for (int i = 0; i < Dimensions.X; i++)
            {
                for (int k = 0; k < Dimensions.Y - 1; k++)
                {
                    for (int j = 0; j < Dimensions.Z; j++)
                    {

                        allNodes[i, k, j] = new Node(new Vector3Int(i, k, j), int.MaxValue);
                        allNodes[i, k, j].Type = _RLLlevel[i, k, j];
                    }
                }
            }

            
            timer.Start();
            
            var startNode = allNodes[Start.X, Start.Y, Start.Z];
            OpenSet.Add(startNode);
            GScore[startNode] = 0;
            FScore[startNode] = getHeuristic(new Vector3(startNode.Position.X, startNode.Position.Y, startNode.Position.Z));

            while (OpenSet.Count > 0)
            {
                var currentNode = FScore.Where(x => OpenSet.Contains(x.Key)).OrderBy(x => x.Value).FirstOrDefault().Key;

                OpenSet.Remove(currentNode);

                if (currentNode.Position.X == Goal.x && currentNode.Position.Y == Goal.y && currentNode.Position.Z == Goal.z)
                {
                    return BackTrack(currentNode);
                }

                ClosedSet.Add(currentNode);

                var neighbours = findNeigbours(currentNode);

                foreach (var neighbour in neighbours)
                {
                    if (ClosedSet.Contains(neighbour)) continue;

                    var tentative_g = GScore[currentNode] + 1; //always distance of 1 to neigbour

                    var neighbourG = GScore.ContainsKey(neighbour) ? GScore[neighbour] : 10_000_000;

                    if (tentative_g < neighbourG)
                    {
                        CameFrom[neighbour] = currentNode;
                        GScore[neighbour] = tentative_g;
                        FScore[neighbour] = tentative_g + getHeuristic(new Vector3(neighbour.Position.X, neighbour.Position.Y, neighbour.Position.Z));

                        if(!OpenSet.Contains(neighbour))
                        {
                            OpenSet.Add(neighbour);
                        }
                    }
                }
            }
            timer.Stop();
            return new List<Node>();
            
        }

        List<Node> findNeigbours(Node currentNode)
        {
            var neighbours = new List<Node>();

            // if (TryGetNode(currentNode.Position.X - 1, currentNode.Position.Y, currentNode.Position.Z, out var left))  openList.Add(new Node(left.Position,currentNode.Cost + 1));
            // if (TryGetNode(currentNode.Position.X + 1, currentNode.Position.Y, currentNode.Position.Z, out var right)) openList.Add(new Node(right.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z - 1, out var front)) openList.Add(new Node(front.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z + 1, out var back)) openList.Add(new Node(back.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y -1, currentNode.Position.Z , out var bottom)) openList.Add(new Node(bottom.Position, currentNode.Cost + 1));
            //if (TryGetNode(currentNode.Position.X, currentNode.Position.Y +1 , currentNode.Position.Z , out var top)) openList.Add(new Node(top.Position, currentNode.Cost + 1));

            if (TryGetNode(currentNode.Position.X - 1, currentNode.Position.Y, currentNode.Position.Z, out var left))
            {
                if (left.Type == 12 || left.Type == 13)
                    //if (left.Cost > currentNode.Cost + 1)
                    {
                        left.Source = currentNode;
                        left.Cost = currentNode.Cost + 1;
                        left.Heuristic = getHeuristic(new Vector3(left.Position.X, left.Position.Y, left.Position.Z));
                    neighbours.Add(left);
                    };

            }
            if (TryGetNode(currentNode.Position.X + 1, currentNode.Position.Y, currentNode.Position.Z, out var right))
            {
                if (right.Type == 12 || right.Type == 13)
                  //  if (right.Cost > currentNode.Cost + 1)
                    {
                        right.Source = currentNode;
                        right.Cost = currentNode.Cost + 1;
                        right.Heuristic = getHeuristic(new Vector3(right.Position.X, right.Position.Y, right.Position.Z));
                    neighbours.Add(right);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z - 1, out var back))
            {
                if (back.Type == 12 || back.Type == 13)
                    //if (back.Cost > currentNode.Cost + 1)
                    {
                        back.Source = currentNode;
                        back.Cost = currentNode.Cost + 1;
                        back.Heuristic = getHeuristic(new Vector3(back.Position.X, back.Position.Y, back.Position.Z));
                    neighbours.Add(back);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y, currentNode.Position.Z + 1, out var front))
            {
                if (front.Type == 12 || front.Type == 13)
                  //  if (front.Cost > currentNode.Cost + 1)
                    {
                        front.Source = currentNode;
                        front.Cost = currentNode.Cost + 1;
                        front.Heuristic = getHeuristic(new Vector3(front.Position.X, front.Position.Y, front.Position.Z));
                    neighbours.Add(front);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y - 1, currentNode.Position.Z, out var bottom))
            {
                if (bottom.Type == 13)
                 //   if (bottom.Cost > currentNode.Cost + 1)
                    {
                        bottom.Source = currentNode;
                        bottom.Cost = currentNode.Cost + 1;
                        bottom.Heuristic = getHeuristic(new Vector3(bottom.Position.X, bottom.Position.Y, bottom.Position.Z));
                    neighbours.Add(bottom);
                    }

            }
            if (TryGetNode(currentNode.Position.X, currentNode.Position.Y + 1, currentNode.Position.Z, out var top))
            {
                if (top.Type == 13)
               //     if (top.Cost > currentNode.Cost + 1)
                    {
                        top.Source = currentNode;
                        top.Cost = currentNode.Cost + 1;
                        top.Heuristic = getHeuristic(new Vector3(top.Position.X,top.Position.Y,top.Position.Z));
                        neighbours.Add(top);
                    }

            }
            //Kosten wenn günstiger ersetzten
            //checken ob schon besucht

            return neighbours;
        }

        float getHeuristic(Vector3 pos)
        {
            Vector3 vec = Goal - pos;
            return vec.magnitude;
        }

        bool TryGetNode(Vector3Int Pos, out Node node)
        {
            node = default;

            if (Pos.X < 0 || Pos.X >= Width) return false;
            if (Pos.Y < 0 || Pos.Y >= Height - 1) return false;
            if (Pos.Z < 0 || Pos.Z >= Depth) return false;

            node = allNodes[Pos.X, Pos.Y, Pos.Z];
            return true;
        }

        bool TryGetNode(int x, int y, int z, out Node node)
        {
            node = default;

            if (x < 0 || x >= Width) return false;
            if (y < 0 || y >= Height - 1) return false;
            if (z < 0 || z >= Depth) return false;

            node = allNodes[x, y, z];
            return true;
        }
    }
}
