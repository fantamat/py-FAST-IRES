import vtk

def create_colored_cube():
    """
    Creates a cube with edges colored by scalar values using the default colormap.
    """
    # Create points for the cube vertices
    points = vtk.vtkPoints()
    points.InsertNextPoint(0, 0, 0)  # Vertex 0
    points.InsertNextPoint(1, 0, 0)  # Vertex 1
    points.InsertNextPoint(1, 1, 0)  # Vertex 2
    points.InsertNextPoint(0, 1, 0)  # Vertex 3
    points.InsertNextPoint(0, 0, 1)  # Vertex 4
    points.InsertNextPoint(1, 0, 1)  # Vertex 5
    points.InsertNextPoint(1, 1, 1)  # Vertex 6
    points.InsertNextPoint(0, 1, 1)  # Vertex 7

    # Create cells (edges) for the cube
    lines = vtk.vtkCellArray()
    
    # Define the 12 edges of the cube
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face edges
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face edges
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    # Create a scalar array for edge values
    edge_scalars = vtk.vtkFloatArray()
    edge_scalars.SetName("EdgeValues")
    
    # Add each edge and assign a scalar value
    for i, (start, end) in enumerate(edges):
        line = vtk.vtkLine()
        line.GetPointIds().SetId(0, start)
        line.GetPointIds().SetId(1, end)
        lines.InsertNextCell(line)
        
        # Assign scalar value to the edge (here using edge index as value)
        # You can replace this with your own values
        edge_scalars.InsertNextValue(i)
    
    # Create a polydata to store the cube
    cube = vtk.vtkPolyData()
    cube.SetPoints(points)
    cube.SetLines(lines)
    
    # Assign the scalar values to the edges
    cube.GetCellData().SetScalars(edge_scalars)
    
    # Create a mapper and actor
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(cube)
    
    # Set scalar range for color mapping (optional)
    mapper.SetScalarRange(0, 11)  # Range from 0 to 11 (number of edges - 1)
    mapper.SetColorModeToMapScalars()  # Map scalars to colors
    
    # Create an actor to represent the cube
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    
    # Make the lines thicker for better visibility
    actor.GetProperty().SetLineWidth(4.0)
    
    # Set up the renderer, window, and interactor
    renderer = vtk.vtkRenderer()
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    
    # Add an interactor for UI control
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Add the actor to the scene
    renderer.AddActor(actor)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue background
    
    # Set up camera
    renderer.ResetCamera()
    
    # Render the scene
    render_window.SetSize(800, 600)
    render_window.Render()
    
    # Start interaction
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    create_colored_cube()
