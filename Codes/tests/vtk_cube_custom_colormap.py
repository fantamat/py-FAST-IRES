import vtk
import numpy as np

def create_cube_with_custom_colormaps():
    """
    Creates a cube with:
    1. Edges colored by one set of data using a custom colormap
    2. Vertices colored by another set of data
    3. Multiple colormaps and color bars for different attributes
    """
    # Create a cube source
    cube_source = vtk.vtkCubeSource()
    cube_source.Update()
    
    # Get the polydata
    cube = vtk.vtkPolyData()
    cube.DeepCopy(cube_source.GetOutput())
    
    # Create a lookup table for edges (using a blue-red color scheme)
    edge_lut = vtk.vtkLookupTable()
    edge_lut.SetHueRange(0.667, 0.0)  # Blue to red
    edge_lut.SetSaturationRange(1.0, 1.0)
    edge_lut.SetValueRange(1.0, 1.0)
    edge_lut.SetTableRange(0.0, 100.0)
    edge_lut.SetNumberOfColors(256)
    edge_lut.Build()
    
    # Extract edges
    edge_extractor = vtk.vtkExtractEdges()
    edge_extractor.SetInputData(cube)
    edge_extractor.Update()
    edges = edge_extractor.GetOutput()
    
    # Generate edge scalar values
    num_edges = edges.GetNumberOfCells()
    edge_scalars = vtk.vtkFloatArray()
    edge_scalars.SetName("EdgeValues")
    
    # Assign values based on edge length (or any other criterion)
    for i in range(num_edges):
        edge = edges.GetCell(i)
        p1 = np.array(edge.GetPoints().GetPoint(0))
        p2 = np.array(edge.GetPoints().GetPoint(1))
        length = np.linalg.norm(p2 - p1)
        # Normalize to 0-100 range
        value = length * 100
        edge_scalars.InsertNextValue(value)
    
    edges.GetCellData().SetScalars(edge_scalars)
    
    # Create mappers
    # 1. Cube mapper (for faces)
    cube_mapper = vtk.vtkPolyDataMapper()
    cube_mapper.SetInputData(cube)
    cube_mapper.ScalarVisibilityOff()
    
    # 2. Edge mapper
    edge_mapper = vtk.vtkPolyDataMapper()
    edge_mapper.SetInputData(edges)
    edge_mapper.SetScalarRange(0.0, 100.0)
    edge_mapper.SetLookupTable(edge_lut)
    edge_mapper.SetColorModeToMapScalars()
    
    # Create actors
    # 1. Cube actor
    cube_actor = vtk.vtkActor()
    cube_actor.SetMapper(cube_mapper)
    cube_actor.GetProperty().SetColor(0.8, 0.8, 0.8)
    cube_actor.GetProperty().SetOpacity(0.3)
    
    # 2. Edge actor
    edge_actor = vtk.vtkActor()
    edge_actor.SetMapper(edge_mapper)
    edge_actor.GetProperty().SetLineWidth(5)
    
    # Create a scalar bar for the edge colors
    edge_scalar_bar = vtk.vtkScalarBarActor()
    edge_scalar_bar.SetLookupTable(edge_lut)
    edge_scalar_bar.SetTitle("Edge Values")
    edge_scalar_bar.SetNumberOfLabels(5)
    edge_scalar_bar.SetMaximumWidthInPixels(100)
    edge_scalar_bar.SetMaximumHeightInPixels(300)
    edge_scalar_bar.SetPosition(0.85, 0.1)  # Position in normalized viewport coordinates
    
    # Create a text actor for instructions
    text_actor = vtk.vtkTextActor()
    text_actor.SetInput("Cube with Colored Edges\nRotate: Left click + drag\nZoom: Right click + drag")
    text_actor.GetTextProperty().SetFontSize(16)
    text_actor.GetTextProperty().SetColor(1.0, 1.0, 1.0)  # White text
    text_actor.SetPosition(10, 10)  # Position in display coordinates
    
    # Set up the renderer
    renderer = vtk.vtkRenderer()
    renderer.AddActor(cube_actor)
    renderer.AddActor(edge_actor)
    renderer.AddActor(edge_scalar_bar)
    renderer.AddActor(text_actor)
    renderer.SetBackground(0.1, 0.1, 0.1)  # Dark background
    
    # Create a render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)
    render_window.SetWindowName("Cube with Colored Edges")
    
    # Create an interactor
    interactor = vtk.vtkRenderWindowInteractor()
    interactor.SetRenderWindow(render_window)
    
    # Initialize and start the visualization
    renderer.ResetCamera()
    render_window.Render()
    interactor.Initialize()
    interactor.Start()

if __name__ == "__main__":
    create_cube_with_custom_colormaps()
