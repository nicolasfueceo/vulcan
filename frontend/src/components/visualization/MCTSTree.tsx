'use client'

import React, { useCallback, useEffect, useMemo } from 'react'
import ReactFlow, {
  Node,
  Edge,
  useNodesState,
  useEdgesState,
  Controls,
  Background,
  MiniMap,
  ConnectionMode,
  ReactFlowProvider
} from 'reactflow'
import dagre from 'dagre'
import { Badge } from '@/components/ui/badge'
import { Card, CardContent } from '@/components/ui/card'
import { MCTSData, MCTSNode as MCTSNodeType } from '@/types/vulcan'

import 'reactflow/dist/style.css'

interface MCTSTreeProps {
  data: MCTSData | null
}

// Custom node component for MCTS nodes
const MCTSNodeComponent = ({ data }: { data: any }) => {
  const getNodeColor = (actionType: string, isSelected: boolean) => {
    if (isSelected) return 'bg-gradient-to-r from-green-400 to-green-600 text-white'
    if (actionType === 'explore') return 'bg-gradient-to-r from-blue-400 to-blue-600 text-white'
    if (actionType === 'exploit') return 'bg-gradient-to-r from-orange-400 to-orange-600 text-white'
    return 'bg-gradient-to-r from-gray-400 to-gray-600 text-white'
  }

  const getUCBColor = (ucbValue: number) => {
    if (ucbValue > 0.8) return 'text-green-600'
    if (ucbValue > 0.5) return 'text-yellow-600'
    return 'text-red-600'
  }

  return (
    <Card className={`min-w-[200px] ${getNodeColor(data.action_taken, data.is_selected)} border-2 shadow-lg hover:shadow-xl transition-shadow`}>
      <CardContent className="p-3">
        <div className="space-y-2">
          {/* Node ID and Action */}
          <div className="flex items-center justify-between">
            <Badge variant="outline" className="text-xs">
              {data.action_taken}
            </Badge>
            <span className="text-xs font-mono">{data.id.slice(0, 8)}...</span>
          </div>

          {/* Score */}
          {data.score !== undefined && (
            <div className="text-center">
              <div className="text-lg font-bold">{data.score.toFixed(4)}</div>
              <div className="text-xs opacity-80">Score</div>
            </div>
          )}

          {/* MCTS Stats */}
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="text-center">
              <div className="font-semibold">{data.visits}</div>
              <div className="opacity-80">Visits</div>
            </div>
            <div className="text-center">
              <div className={`font-semibold ${getUCBColor(data.ucb_value)}`}>
                {data.ucb_value.toFixed(3)}
              </div>
              <div className="opacity-80">UCB</div>
            </div>
          </div>

          {/* Feature Name */}
          {data.feature_name && (
            <div className="text-xs text-center">
              <div className="font-semibold truncate" title={data.feature_name}>
                {data.feature_name}
              </div>
            </div>
          )}

          {/* Depth and Expansion Status */}
          <div className="flex items-center justify-between text-xs">
            <span>D:{data.depth}</span>
            <div className="flex gap-1">
              {data.is_terminal && <Badge variant="destructive" className="text-xs">T</Badge>}
              {data.is_fully_expanded && <Badge variant="secondary" className="text-xs">F</Badge>}
            </div>
          </div>

          {/* Features Count */}
          {data.cumulative_features && (
            <div className="text-xs text-center">
              <span className="font-semibold">{data.cumulative_features.length}</span> features
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}

// Layout nodes using dagre
const getLayoutedElements = (nodes: Node[], edges: Edge[], direction = 'TB') => {
  const isHorizontal = direction === 'LR'
  const dagreGraph = new dagre.graphlib.Graph()
  dagreGraph.setDefaultEdgeLabel(() => ({}))
  dagreGraph.setGraph({ rankdir: direction, nodesep: 100, ranksep: 150 })

  nodes.forEach((node) => {
    dagreGraph.setNode(node.id, { width: 220, height: 160 })
  })

  edges.forEach((edge) => {
    dagreGraph.setEdge(edge.source, edge.target)
  })

  dagre.layout(dagreGraph)

  return {
    nodes: nodes.map((node) => {
      const nodeWithPosition = dagreGraph.node(node.id)
      return {
        ...node,
        targetPosition: isHorizontal ? 'left' : 'top',
        sourcePosition: isHorizontal ? 'right' : 'bottom',
        position: {
          x: nodeWithPosition.x - 110,
          y: nodeWithPosition.y - 80,
        },
      }
    }),
    edges,
  }
}

export function MCTSTree({ data }: MCTSTreeProps) {
  const [nodes, setNodes, onNodesChange] = useNodesState([])
  const [edges, setEdges, onEdgesChange] = useEdgesState([])

  // Convert MCTS data to ReactFlow format
  const { flowNodes, flowEdges } = useMemo(() => {
    if (!data || !data.nodes.length) {
      return { flowNodes: [], flowEdges: [] }
    }

    const flowNodes: Node[] = data.nodes.map((node: MCTSNodeType) => ({
      id: node.id,
      type: 'default',
      data: {
        ...node,
        is_selected: node.id === data.best_node_id,
        label: node.feature_name || `Node ${node.id.slice(0, 8)}`
      },
      position: { x: 0, y: 0 }, // Will be set by layout
    }))

    const flowEdges: Edge[] = data.edges.map((edge) => ({
      id: edge.id,
      source: edge.source,
      target: edge.target,
      type: 'smoothstep',
      animated: true,
      style: {
        stroke: edge.action_type === 'explore' ? '#3b82f6' : '#f97316',
        strokeWidth: 2,
      },
      label: edge.action_type,
      labelStyle: {
        fill: edge.action_type === 'explore' ? '#3b82f6' : '#f97316',
        fontWeight: 600,
        fontSize: 12,
      },
    }))

    return { flowNodes, flowEdges }
  }, [data])

  // Apply layout and update nodes/edges
  useEffect(() => {
    if (flowNodes.length > 0) {
      const { nodes: layoutedNodes, edges: layoutedEdges } = getLayoutedElements(
        flowNodes,
        flowEdges
      )
      setNodes(layoutedNodes)
      setEdges(layoutedEdges)
    } else {
      setNodes([])
      setEdges([])
    }
  }, [flowNodes, flowEdges, setNodes, setEdges])

  // Custom node types
  const nodeTypes = useMemo(
    () => ({
      default: MCTSNodeComponent,
    }),
    []
  )

  if (!data || !data.nodes.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">🌳</div>
          <p>No MCTS tree data available</p>
          <p className="text-sm">Start an MCTS experiment to see the search tree</p>
        </div>
      </div>
    )
  }

  return (
    <div className="w-full h-[600px] border rounded-lg">
      <ReactFlowProvider>
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          connectionMode={ConnectionMode.Loose}
          fitView
          fitViewOptions={{ padding: 0.2 }}
        >
          <Controls />
          <Background />
          <MiniMap 
            nodeColor={(node) => {
              if (node.data?.is_selected) return '#22c55e'
              if (node.data?.action_taken === 'explore') return '#3b82f6'
              if (node.data?.action_taken === 'exploit') return '#f97316'
              return '#6b7280'
            }}
            maskColor="rgb(240, 240, 240, 0.6)"
          />
        </ReactFlow>
      </ReactFlowProvider>

      {/* Tree Statistics */}
      <div className="mt-4 grid grid-cols-5 gap-4 text-center">
        <div>
          <div className="text-lg font-bold text-blue-600">{data.stats.total_nodes}</div>
          <div className="text-xs text-slate-600">Total Nodes</div>
        </div>
        <div>
          <div className="text-lg font-bold text-green-600">{data.stats.max_depth}</div>
          <div className="text-xs text-slate-600">Max Depth</div>
        </div>
        <div>
          <div className="text-lg font-bold text-purple-600">{data.stats.best_score.toFixed(4)}</div>
          <div className="text-xs text-slate-600">Best Score</div>
        </div>
        <div>
          <div className="text-lg font-bold text-orange-600">{data.stats.iterations_completed}</div>
          <div className="text-xs text-slate-600">Iterations</div>
        </div>
        <div>
          <div className="text-lg font-bold text-gray-600">{data.stats.avg_branching_factor.toFixed(1)}</div>
          <div className="text-xs text-slate-600">Avg Branching</div>
        </div>
      </div>
    </div>
  )
} 