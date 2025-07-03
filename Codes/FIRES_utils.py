import numpy as np
import os
import vtk
import matplotlib.pyplot as plt

def upsample(x, n):
    # Insert (n - 1) zeros between rows (like MATLAB)
    out = np.zeros((x.shape[0] * n, x.shape[1]), dtype=x.dtype)
    out[::n] = x
    return out

def circshift(arr, shift):
    # Equivalent to MATLAB's circshift
    return np.roll(arr, shift, axis=(0, 1))


def readlocs(filename):
    # Reads electrode locations from a .xyz file (tab-separated)
    # Returns a numpy array of shape (n_electrodes, 4): [x, y, z, label]
    positions = []
    labels = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            idx, x, y, z, label = parts[:5]
            positions.append([float(x), float(y), float(z)])
            labels.append(label.strip())
    positions = np.array(positions)
    labels = np.array(labels)
    return positions, labels


def create_custom_colormap():
    """
    Creates a custom colormap similar to the one in the MATLAB code:
    cmap1 = jet(64);
    part1 = cmap1(1:31,:);
    part2 = cmap1(34:end,:);
    mid_tran = gray(64);
    mid = mid_tran(57:58,:);
    cmap = [part1;mid;part2];
    """
    # Create a jet colormap with 64 colors
    jet_cmap = plt.cm.jet
    jet_colors = jet_cmap(np.linspace(0, 1, 64))
    
    # Get the parts as specified in MATLAB
    part1 = jet_colors[:31, :]
    part2 = jet_colors[33:, :]
    
    # Create a gray colormap with 64 colors
    gray_cmap = plt.cm.gray
    gray_colors = gray_cmap(np.linspace(0, 1, 64))
    mid = gray_colors[56:58, :]
    
    # Combine the parts
    custom_colors = np.vstack((part1, mid, part2))
    
    # Create a new colormap
    custom_cmap = plt.cm.colors.ListedColormap(custom_colors)
    
    return custom_cmap

def plot_mesh(curryloc, currytri, Ab_J, J, weight_it, Fig_Folder, perspect):
    """    Plots the mesh surface with trisurf and colors based on the sum of J values.
    Args:
        curryloc (np.ndarray): Current locations of vertices (3 x n).
        currytri (np.ndarray): Triangle indices for the mesh (m x 3).
        Ab_J (np.ndarray): Absolute values of J (n x 1).
        J (np.ndarray): Current J values (n x m).
        weight_it (int): Current iteration number for naming the figure.
        Fig_Folder (str): Folder to save the figure.
        perspect (list): Perspective settings for the plot.
    """
    # Create figure with the same size as MATLAB
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for trisurf
    x = curryloc[0, :]
    y = curryloc[1, :]
    z = curryloc[2, :]
    facecolors = np.sum(Ab_J, axis=1)

    # Plot trisurf - use similar settings to MATLAB's trisurf
    # Note: MATLAB indices start at 1, so subtract 1 for Python
    trisurf = ax.plot_trisurf(x, y, z, triangles=currytri, cmap=create_custom_colormap(), 
                              linewidth=0, antialiased=True, shade=True)
    
    # Set facecolor data similar to MATLAB
    trisurf.set_array(facecolors)
    
    # Add a colorbar like in MATLAB
    cbar = fig.colorbar(trisurf, ax=ax, shrink=0.5, aspect=10)
    
    # Set the view to match MATLAB's view(perspect)
    ax.view_init(elev=perspect[1]*90, azim=perspect[0]*90)
    
    # Turn off the grid like in MATLAB
    ax.grid(False)
    
    # Set color limits similar to MATLAB's caxis
    x_max = np.max(np.abs(np.sum(J, axis=1)))
    if np.isnan(x_max):
        x_max = 1
    trisurf.set_clim(-x_max, x_max)
    
    # Add lighting similar to MATLAB
    # Note: Matplotlib's lighting is more limited than MATLAB's
    # but we can approximate it
    from matplotlib.colors import LightSource
    ls = LightSource(azdeg=45, altdeg=45)
    
    # Set other properties to match MATLAB
    ax.set_facecolor('white')  # Background color
    
    # Apply similar phong shading and lighting as in MATLAB
    # This is an approximation as matplotlib doesn't have direct equivalents
    plt.tight_layout()
    
    # Save figures in the same format as MATLAB
    if not os.path.exists(Fig_Folder):
        os.makedirs(Fig_Folder)
        
    fig_name_fig = os.path.join(Fig_Folder, f'TBF_1st_Iteration_{weight_it}.png')
    plt.savefig(fig_name_fig, dpi=300, bbox_inches='tight')
    fig_name_fig_jpeg = os.path.join(Fig_Folder, f'TBF_1st_Iteration_{weight_it}.jpeg')
    plt.savefig(fig_name_fig_jpeg, dpi=300, bbox_inches='tight')
    plt.show()  # Optional: display the plot interactively if needed    
    plt.close(fig)
    

def create_jet_mid_jet_lut():
    """
    Creates a custom lookup table for VTK that mimics the MATLAB colormap:
    cmap1 = jet(64);
    part1 = cmap1(1:31,:);
    part2 = cmap1(34:end,:);
    mid_tran = gray(64);
    mid = mid_tran(57:58,:);
    cmap = [part1;mid;part2];
    """
    # Create a custom color map similar to MATLAB's
    lut = vtk.vtkLookupTable()
    lut.SetNumberOfTableValues(64)
    lut.Build()
    
    # Create a jet colormap with 64 colors
    jet_cmap = plt.cm.jet
    jet_colors = jet_cmap(np.linspace(0, 1, 64))
    
    # Get the parts as specified in MATLAB
    part1 = jet_colors[:31, :]
    part2 = jet_colors[33:, :]
    
    # Create a gray colormap with 64 colors
    gray_cmap = plt.cm.gray
    gray_colors = gray_cmap(np.linspace(0, 1, 64))
    mid = gray_colors[56:58, :]
    
    # Combine the parts
    custom_colors = np.vstack((part1, mid, part2))
    
    # Set the colors in the lookup table
    for i in range(custom_colors.shape[0]):
        r, g, b, a = custom_colors[i]
        lut.SetTableValue(i, r, g, b, 1.0)  # Alpha is 1.0 (fully opaque)
    
    return lut

def plot_mesh_vtk(curryloc, currytri, Ab_J, J, weight_it, Fig_Folder, perspect):
    """
    Plot mesh using VTK with settings similar to MATLAB's trisurf.
    """
    # Create directory if it doesn't exist
    if not os.path.exists(Fig_Folder):
        os.makedirs(Fig_Folder)
    
    # Step 1: Create VTK points
    points = vtk.vtkPoints()
    n_points = curryloc.shape[1]
    for i in range(n_points):
        points.InsertNextPoint(curryloc[0, i], curryloc[1, i], curryloc[2, i])

    # Step 2: Create VTK triangles
    triangles = vtk.vtkCellArray()
    for tri in currytri[:len(currytri)]:
        triangle = vtk.vtkTriangle()
        triangle.GetPointIds().SetId(0, tri[0])
        triangle.GetPointIds().SetId(1, tri[1])
        triangle.GetPointIds().SetId(2, tri[2])
        triangles.InsertNextCell(triangle)

    # Step 3: Create polydata mesh
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(points)
    polydata.SetPolys(triangles)

    # Step 4: Compute color values (based on Ab_J sum)
    facecolors = np.sum(Ab_J, axis=1)
    vtk_colors = vtk.vtkFloatArray()
    vtk_colors.SetName("Colors")
    for i in range(n_points):
        contains_i = np.where(np.any(currytri == i, axis=1))
        val = facecolors[contains_i].mean() if contains_i[0].size > 0 else 0.0
        vtk_colors.InsertNextValue(val)
    polydata.GetPointData().SetScalars(vtk_colors)

    # Step 5: Set up mapper with custom LUT (similar to MATLAB's colormap)
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputData(polydata)
    
    # Set color limits similar to MATLAB's caxis
    x_max = np.max(np.abs(np.sum(J, axis=1)))
    if np.isnan(x_max):
        x_max = 1
    mapper.SetScalarRange(-x_max, x_max)
    
    # Use our custom lookup table
    mapper.SetLookupTable(create_jet_mid_jet_lut())
    mapper.SetColorModeToMapScalars()
    mapper.SetScalarModeToUsePointData()

    # Step 6: Actor with properties similar to MATLAB's trisurf settings
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(1.0)  # Full opacity, like MATLAB
    actor.GetProperty().SetSpecular(0.3)  # Add some specularity for phong lighting
    actor.GetProperty().SetSpecularPower(20)
    actor.GetProperty().SetInterpolationToPhong()  # Phong shading, like MATLAB
    # Make edges invisible
    actor.GetProperty().EdgeVisibilityOff()
    
    # Step 7: Renderer with white background
    renderer = vtk.vtkRenderer()
    renderer.AddActor(actor)
    renderer.SetBackground(1, 1, 1)  # White background

    # Add lighting similar to MATLAB
    # MATLAB has: light('Position',[3 3 1]) and light('Position',[-3 -3 -1])
    light1 = vtk.vtkLight()
    light1.SetPosition(3, 3, 1)
    light1.SetColor(1, 1, 1)
    light1.SetIntensity(0.8)
    renderer.AddLight(light1)
    
    light2 = vtk.vtkLight()
    light2.SetPosition(-3, -3, -1)
    light2.SetColor(1, 1, 1)
    light2.SetIntensity(0.8)
    renderer.AddLight(light2)

    # Set the view to match MATLAB's view(perspect)
    camera = vtk.vtkCamera()
    camera.SetViewUp(0, 0, 1)
    camera.SetPosition(perspect[0] * 90, perspect[1] * 90, 100)
    camera.SetFocalPoint(0, 0, 0)
    renderer.SetActiveCamera(camera)
    renderer.ResetCamera()

    # Add a scalar bar (colorbar) similar to MATLAB
    scalar_bar = vtk.vtkScalarBarActor()
    scalar_bar.SetLookupTable(mapper.GetLookupTable())
    scalar_bar.SetNumberOfLabels(5)
    scalar_bar.SetTitle("Value")
    scalar_bar.SetLabelFormat("%.1f")
    scalar_bar.SetPosition(0.9, 0.1)
    scalar_bar.SetPosition2(0.1, 0.8)
    renderer.AddActor2D(scalar_bar)

    # Step 8: Render window
    render_window = vtk.vtkRenderWindow()
    render_window.AddRenderer(renderer)
    render_window.SetSize(800, 600)  # Similar size to MATLAB figures

    # Step 9: Off-screen rendering and save
    window_to_image_filter = vtk.vtkWindowToImageFilter()
    window_to_image_filter.SetInput(render_window)
    window_to_image_filter.ReadFrontBufferOff()
    window_to_image_filter.Update()

    # Render and save as both PNG and JPEG
    render_window.Render()
    
    # Save PNG
    writer = vtk.vtkPNGWriter()
    writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    file_path = os.path.join(Fig_Folder, f'VTK_TBF_1st_Iteration_{weight_it}.png')
    writer.SetFileName(file_path)
    writer.Write()

    # Save JPEG
    jpeg_writer = vtk.vtkJPEGWriter()
    jpeg_writer.SetInputConnection(window_to_image_filter.GetOutputPort())
    file_path_jpeg = os.path.join(Fig_Folder, f'VTK_TBF_1st_Iteration_{weight_it}.jpeg')
    jpeg_writer.SetFileName(file_path_jpeg)
    jpeg_writer.Write()

    # Optionally display the visualization interactively
    VTK_SHOW = True  # Set to True if you want interactive visualization
    if VTK_SHOW:
        # Add interactor for interactive viewing
        render_window_interactor = vtk.vtkRenderWindowInteractor()
        render_window_interactor.SetRenderWindow(render_window)
        render_window.Render()
        render_window_interactor.Initialize()
        render_window_interactor.Start()
