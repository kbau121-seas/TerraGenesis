#import maya.api.OpenMaya as om
import maya.OpenMaya as om
import maya.OpenMayaMPx as ompx
import maya.cmds as cmds
import math
import numpy as np
from PIL import Image

# Utility functions for setting attribute flags
def setInputAttr(attr):
    attr.keyable = True
    attr.storable = True
    attr.readable = True
    attr.writable = True

def setOutputAttr(attr):
    attr.keyable = False
    attr.storable = False
    attr.readable = True
    attr.writable = False

class TerrainGenTask1Node(ompx.MPxNode):
    # Define node name and unique ID
    kNodeName = "TerrainGenTask1Node"
    kNodeId   = om.MTypeId(0x00082000)

    # Attributes
    aUpliftPath = None    # String attribute: uplift image file path
    aTimeStep   = None    # Float attribute: simulation time step
    aIterations = None    # Int attribute: number of simulation iterations
    aGridSizeX  = None    # Float attribute: physical grid size in X direction
    aGridSizeZ  = None    # Float attribute: physical grid size in Z direction
    aCellSize   = None    # Float attribute: size of each cell in the grid
    aMeshOutput = None    # Mesh attribute: output terrain mesh
    

    def __init__(self):
        super(TerrainGenTask1Node, self).__init__()

    def compute(self, plug, dataBlock):
        # Only compute the output mesh
        if plug != TerrainGenTask1Node.aMeshOutput and plug.parent() != TerrainGenTask1Node.aMeshOutput:
            om.MGlobal.displayInfo("Compute function nottriggered")
            return

        # Retrieve input attribute values
        om.MGlobal.displayInfo("Compute function triggered")
        upliftPath = dataBlock.inputValue(TerrainGenTask1Node.aUpliftPath).asString()
        timeStep   = dataBlock.inputValue(TerrainGenTask1Node.aTimeStep).asFloat()
        iterations = dataBlock.inputValue(TerrainGenTask1Node.aIterations).asInt()
        gridX      = dataBlock.inputValue(TerrainGenTask1Node.aGridSizeX).asFloat()
        gridZ      = dataBlock.inputValue(TerrainGenTask1Node.aGridSizeZ).asFloat()
        cellSize   = dataBlock.inputValue(TerrainGenTask1Node.aCellSize).asFloat()

        # Calculate grid dimensions (ensure a minimum grid of 4x4 cells)
        rows = max(int(math.ceil(gridX / cellSize)), 4)
        cols = max(int(math.ceil(gridZ / cellSize)), 4)
        gridDims = (rows, cols)

        # Load the uplift image and resize it to match the grid dimensions
        upliftArray = self.loadUpliftImage(upliftPath, gridDims)

        # Initialize the elevation array with zeros
        elevation = np.zeros(gridDims, dtype=np.float32)

        # Run a basic simulation: for each iteration, add uplift scaled by the time step
        for i in range(iterations):
            elevation += upliftArray * timeStep

       

        # Set the computed mesh as the output
        outHandle:om.MDataHandle = dataBlock.outputValue(TerrainGenTask1Node.aMeshOutput)
        meshData:om.MObject = om.MFnMeshData().create()
        om.MGlobal.displayInfo("meshData created, type: " + str(type(meshData)))
        self.buildMesh(elevation, cellSize,meshData)
        outHandle.setMObject(meshData)
        dataBlock.setClean(plug)
        

    def loadUpliftImage(self, path, dims):
        """
        Loads an image from 'path', converts it to grayscale,
        normalizes the pixel values to [0,1], and resizes it to 'dims'.
        """
        if not path:
            om.MGlobal.displayWarning("Uplift path is empty.")
            return np.zeros(dims, dtype=np.float32)
        try:
            with Image.open(path) as img:
                om.MGlobal.displayInfo(f"Uplift image loaded from {path}")
                # Convert the image to grayscale ('L' mode)
                img = img.convert("L")
                imgArray = np.array(img, dtype=np.float32) / 255.0
                # Resize the image to the desired dimensions (PIL expects (width, height))
                resizedImg = Image.fromarray((imgArray * 255).astype(np.uint8)).resize((dims[1], dims[0]), Image.BILINEAR)
                resizedArray = np.array(resizedImg, dtype=np.float32) / 255.0
                return resizedArray
        except Exception as e:
            om.MGlobal.displayWarning("Error loading uplift image: " + str(e))
            return np.zeros(dims, dtype=np.float32)

    def buildMesh(self, elevation, cellSize,meshdata:om.MObject):
        """
        Constructs a polygon mesh based on the elevation data.
        The grid is centered at the origin on the XZ-plane, and each vertex's Y coordinate
        corresponds to the elevation value.
        """
        om.MGlobal.displayInfo(f"Building mesh with dimensions: {elevation.shape}")
        #om.MGlobal.displayInfo("meshdata type: " + str(type(meshdata)))
        rows, cols = elevation.shape
        vertices = []
        polyCounts = []
        polyConnects = []
        # Center the grid on the origin
        originX = -0.5 * cellSize * (rows - 1)
        originZ = -0.5 * cellSize * (cols - 1)
        indexMap = {}

        # Create vertices from the elevation grid
        for i in range(rows):
            x = originX + i * cellSize
            for j in range(cols):
                z = originZ + j * cellSize
                y = elevation[i, j]
                vertices.append(om.MPoint(float(x),float(y), float(z)))
                indexMap[(i, j)] = len(vertices) - 1

        # Create quads for each cell in the grid (except the last row and column)
        for i in range(rows - 1):
            for j in range(cols - 1):
                polyCounts.append(4)
                polyConnects.extend([
                    indexMap[(i, j)],
                    indexMap[(i, j+1)],
                    indexMap[(i+1, j+1)],
                    indexMap[(i+1, j)]
                ])

        # Build the mesh using Maya's API
        pointsArray = om.MPointArray()
        for pt in vertices:
            pointsArray.append(pt)

        countsArray = om.MIntArray()
        for count in polyCounts:
            countsArray.append(count)

        connectsArray = om.MIntArray()
        for idx in polyConnects:
            connectsArray.append(int(idx))
        meshFn = om.MFnMesh()
        om.MGlobal.displayInfo(f"=====: {type(pointsArray)}")
        om.MGlobal.displayInfo(f"=====: {type(countsArray)}")
        om.MGlobal.displayInfo(f"=====: {type(connectsArray)}")
        om.MGlobal.displayInfo(f"=====: {type(meshdata)}")
        om.MGlobal.displayInfo(f"=====: {om.MGlobal.apiVersion()}")

        numVerts = pointsArray.length()
        numPolys = countsArray.length()

        meshObj = meshFn.create(numVerts, numPolys, pointsArray, countsArray, connectsArray, meshdata)



        # meshObj:om.MObject= meshFn.create(pointsArray,countsArray, connectsArray, meshdata)

        return meshObj

    @staticmethod
    def creator():
        return TerrainGenTask1Node()

    @staticmethod
    def initialize():
        # Create numeric and typed attribute functions
        numAttrFn = om.MFnNumericAttribute()
        typeAttrFn = om.MFnTypedAttribute()
        strDataFn = om.MFnStringData()

        # Uplift Path attribute (string, used as filename)
        defaultStr = strDataFn.create("")
        TerrainGenTask1Node.aUpliftPath = typeAttrFn.create("upliftFile", "upf", om.MFnData.kString, defaultStr)
        setInputAttr(typeAttrFn)
        typeAttrFn.usedAsFilename = True
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aUpliftPath)

        # Time Step attribute (float)
        TerrainGenTask1Node.aTimeStep = numAttrFn.create("timeStep", "ts", om.MFnNumericData.kFloat, 0.1)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aTimeStep)

        # Iterations attribute (int)
        TerrainGenTask1Node.aIterations = numAttrFn.create("iterations", "iter", om.MFnNumericData.kInt, 10)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aIterations)

        # Grid Size X attribute (float)
        TerrainGenTask1Node.aGridSizeX = numAttrFn.create("gridSizeX", "gsx", om.MFnNumericData.kFloat, 10.0)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aGridSizeX)

        # Grid Size Z attribute (float)
        TerrainGenTask1Node.aGridSizeZ = numAttrFn.create("gridSizeZ", "gsz", om.MFnNumericData.kFloat, 10.0)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.0)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aGridSizeZ)

        # Cell Size attribute (float)
        TerrainGenTask1Node.aCellSize = numAttrFn.create("cellSize", "cs", om.MFnNumericData.kFloat, 0.5)
        setInputAttr(numAttrFn)
        numAttrFn.setMin(0.001)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aCellSize)

        # Output Mesh attribute (mesh data)
        TerrainGenTask1Node.aMeshOutput = typeAttrFn.create("outputMesh", "out", om.MFnData.kMesh)
        setOutputAttr(typeAttrFn)
        ompx.MPxNode.addAttribute(TerrainGenTask1Node.aMeshOutput)

        # Define attribute affects so that changes in any input update the mesh
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aUpliftPath, TerrainGenTask1Node.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aTimeStep, TerrainGenTask1Node.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aIterations, TerrainGenTask1Node.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aGridSizeX, TerrainGenTask1Node.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aGridSizeZ, TerrainGenTask1Node.aMeshOutput)
        ompx.MPxNode.attributeAffects(TerrainGenTask1Node.aCellSize, TerrainGenTask1Node.aMeshOutput)

def create_terrain_node():
    """
    Helper function to create an instance of the TerrainGenTask1Node in the scene.
    If a node with the same name exists, it is deleted first.
    """
    nodeName = "myTerrainNode"
    if cmds.objExists(nodeName):
        cmds.delete(nodeName)
    newNode = cmds.createNode(TerrainGenTask1Node.kNodeName, name=nodeName)
    om.MGlobal.displayInfo("Created node: " + newNode)
    transformName = cmds.createNode("transform", name="terrainTransform#")
    meshShape = cmds.createNode("mesh", parent=transformName, name="terrainMesh#")
    # connect output mesh to inmesh
    cmds.connectAttr(newNode + ".outputMesh", meshShape + ".inMesh", force=True)
    return newNode

# Plugin initialization functions
def initializePlugin(mObject):
    om.MGlobal.displayInfo("Created node: ")
    plugin = ompx.MFnPlugin(mObject, "MyCompany", "1.0", "Any")
    try:
        plugin.registerNode(TerrainGenTask1Node.kNodeName,
                              TerrainGenTask1Node.kNodeId,
                              TerrainGenTask1Node.creator,
                              TerrainGenTask1Node.initialize,
                              ompx.MPxNode.kDependNode)
    except Exception as e:
        om.MGlobal.displayError("Error registering node: " + str(e))
    create_terrain_node()

def uninitializePlugin(mObject):
    plugin = ompx.MFnPlugin(mObject)
    try:
        plugin.deregisterNode(TerrainGenTask1Node.kNodeId)
    except Exception as e:
        om.MGlobal.displayError("Error deregistering node: " + str(e))


