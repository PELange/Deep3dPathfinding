using static Assets.Scripts.VoxelWorld.VoxelWorldGenerator;

namespace Assets.Scripts.VoxelWorld
{
    public class VoxelWorld
    {
        public int Width { get; set; }

        
        public int Height { get; set; }

        public int Depth { get; set; }

        public LevelType levelType;



        public Voxel[,,] Voxels { get; set; }

        public VoxelWorld(int width, int height ,int depth, LevelType level)
        {
            Width = width;
            Height = height;
            Depth = depth;
            levelType = level;

            Voxels = new Voxel[width, height, depth];
        }

        public bool TrySetVoxel(Voxel voxel)
        {
            if (voxel == null) return false;

            if (voxel.Position.X < 0 || voxel.Position.X >= Width) return false;
            if (voxel.Position.Y < 0 || voxel.Position.Y >= Height) return false;
            if (voxel.Position.Z < 0 || voxel.Position.Z >= Depth) return false;

            Voxels[voxel.Position.X, voxel.Position.Y, voxel.Position.Z] = voxel;

            return true;
        }

        public bool TryGetVoxel(int x, int height, int z, out Voxel voxel)
        {
            voxel = default;

            if (x < 0 || x >= Width) return false;
            if (height < 0 || height >= Height) return false;
            if (z < 0 || z >= Depth) return false;

            voxel = Voxels[x, height, z];

            return true;
        }

        public Voxel GetVoxel(int x, int height, int z)
        {
            return Voxels[x, height, z];
        }
    }
}
