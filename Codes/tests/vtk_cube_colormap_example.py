import vtk

def create_cube_with_colored_edges():
    """
    Creates a cube with visible faces and edges colored by scalar values.
    The edges are colored using the default rainbow colormap.
    """
    # Create a cube source
    cube_source = vtk.vtkCubeSource()
    cube_source.SetXLength(1.0)
    cube_source.SetYLength(1.0)
    cube_source.SetZLength(1.0)
    cube_source.SetCenter(0.5, 0.5, 0.5)  # Center at (0.5, 0.5, 0.5)
    cube_source.Update()
    
    # Extract edges from the cube
    edge_extractor = vtk.vtkExtractEdges()
    edge_extractor.SetInputData(cube_source.GetOutput())
    edge_extractor.Update()
    
    # Get the extracted edges as polydata
    edges = edge_extractor.GetOutput()
    
    # Create scalar values for the edges
    num_edges = edges.GetNumberOfCells()
    edge_values = vtk.vtkFloatArray()
    edge_values.SetName("EdgeValues")
    
    # Assign different values to each edge
    for i in range(num_edges):
        # You can customize this to use your own data
        value = i / float(num_edges - 1) * 100  # Scale to 0-100
        edge_values.InsertNextValue(value)
    
    # Assign the scalar values to edge cells
    edges.GetCellData().SetScalars(edge_values)
    
    # Create a mapper for the cube faces
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputData(cube_source.GetOutput())
    cube_mapper.ScalarVisibilityOff()  # Don't color faces by scalars
    
    # Create a mapper for the edges
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputData(edges)
    edge_mapper.SetScalarRange(0, 100)  # Set the range for color mapping
    
    # Create actors
    # 1. Cube actor (semi-transparent)
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetColor(0.8, 0.8, 0.8)  # Light gray
    cube_actor.GetProperty().SetOpacity(0.3)  # Make faces semi-transparent
    
    # 2. Edges actor
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetLineWidth(5)  # Make edges thicker
    
    # Create a color bar (scalar bar) actor
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(edge_mapper.GetLookupTable())
    scalar_bar.SetTitle("Edge Values")
    scalar_bar.SetNumberOfLabels(5)
    
    # Create a renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(cube_actor)
    renderer.AddActor(edge_actor)
    renderer.AddActor(scalar_bar)
    renderer.SetBackground(0.1, 0.2, 0.3)  # Dark blue background
    
    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    
    # Create an interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Initialize and start the visualization
    renderer.ResetCamera()
    render_window.Render()
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    create_cube_with_colored_edges()
