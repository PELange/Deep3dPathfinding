using System.Collections.Generic;
using UnityEngine;

namespace Assets.Scripts.VoxelWorld
{
    public class VoxelSegment
    {
        public int Label { get; set; }

        public List<Voxel> Voxels { get; set; }

        public Color DebugColor { get; set; }

        public VoxelSegment(int labelId)
        {
            Label = labelId;
            Voxels = new List<Voxel>();
        }
    }
}
