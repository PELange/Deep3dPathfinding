namespace Assets.Scripts.VoxelWorld
{
    public class VoxelElevator
    {
        public Voxel ElevatorExitTop { get; set; }
        public Voxel ElevatorEntryBottom { get; set; }
        public int StepsDown { get; set; }

        public VoxelElevator(Voxel elevatorExit, int stepsDown)
        {
            ElevatorExitTop = elevatorExit;
            StepsDown = stepsDown;
        }
    }
}
