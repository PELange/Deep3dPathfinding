namespace Assets.Scripts.VoxelWorld
{
    public class Voxel
    {
        public enum VoxelType
        {
            Empty,
            Solid,
            ElevatorRoot,
            ElevatorShaft,
        }

        public Vector3Int Position { get; set; }

        public VoxelType Type { get; set; }

        public int ElevatorHeight = 0;
        public Voxel(Vector3Int position, VoxelType type)
        {
            Position = position;
            Type = type;
        }

        public Voxel(int x,  int height ,int z, VoxelType type)
        {
            Position = new Vector3Int(x, height, z);
            Type = type;
        }
    }
}
