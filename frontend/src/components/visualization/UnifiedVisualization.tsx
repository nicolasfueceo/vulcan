'use client'

import React from 'react'
import { EvolutionVisualization } from '@/components/evolution-visualization'
import { MCTSTree } from '@/components/visualization/MCTSTree'
import { NarrativeLog } from '@/components/visualization/NarrativeLog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { EvolutionData, MCTSData, VisualizationData } from '@/types/vulcan'

interface UnifiedVisualizationProps {
  data: VisualizationData | null
  algorithm: 'evolution' | 'mcts'
}

// Type guard to check if data is MCTS data
function isMCTSData(data: VisualizationData): data is MCTSData {
  return data && 'nodes' in data && 'edges' in data
}

// Type guard to check if data is Evolution data  
function isEvolutionData(data: VisualizationData): data is EvolutionData {
  return data && 'population' in data && !('nodes' in data)
}

export function UnifiedVisualization({ data, algorithm }: UnifiedVisualizationProps) {
  if (!data) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">
            {algorithm === 'mcts' ? '🌳' : '🧬'}
          </div>
          <p>No {algorithm === 'mcts' ? 'MCTS' : 'evolution'} data available</p>
          <p className="text-sm">
            Start an experiment to see the {algorithm === 'mcts' ? 'search tree' : 'population evolve'}
          </p>
        </div>
      </div>
    )
  }

  // Handle MCTS visualization
  if (algorithm === 'mcts' && isMCTSData(data)) {
    return (
      <Tabs defaultValue="tree" className="space-y-6">
        <TabsList className="grid w-full grid-cols-3">
          <TabsTrigger value="tree">Search Tree</TabsTrigger>
          <TabsTrigger value="narrative">Decision Log</TabsTrigger>
          <TabsTrigger value="overview">Overview</TabsTrigger>
        </TabsList>

        <TabsContent value="tree" className="space-y-6">
          <MCTSTree data={data} />
        </TabsContent>

        <TabsContent value="narrative" className="space-y-6">
          <NarrativeLog 
            decisionLogs={[]} 
            llmInteractions={[]} 
          />
        </TabsContent>

        <TabsContent value="overview" className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* MCTS Statistics */}
            <div className="space-y-4">
              <h3 className="text-lg font-semibold">MCTS Statistics</h3>
              <div className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 bg-blue-50 rounded-lg">
                  <div className="text-2xl font-bold text-blue-600">{data.stats.total_nodes}</div>
                  <div className="text-sm text-slate-600">Total Nodes</div>
                </div>
                <div className="text-center p-4 bg-green-50 rounded-lg">
                  <div className="text-2xl font-bold text-green-600">{data.stats.max_depth}</div>
                  <div className="text-sm text-slate-600">Max Depth</div>
                </div>
                <div className="text-center p-4 bg-purple-50 rounded-lg">
                  <div className="text-2xl font-bold text-purple-600">{data.stats.best_score.toFixed(4)}</div>
                  <div className="text-sm text-slate-600">Best Score</div>
                </div>
                <div className="text-center p-4 bg-orange-50 rounded-lg">
                  <div className="text-2xl font-bold text-orange-600">{data.stats.iterations_completed}</div>
                  <div className="text-sm text-slate-600">Iterations</div>
                </div>
              </div>
            </div>

            {/* Best Path */}
            {data.best_node_id && (
              <div className="space-y-4">
                <h3 className="text-lg font-semibold">Best Path</h3>
                <div className="space-y-2">
                  {/* Find path to best node */}
                  {(() => {
                    const bestNode = data.nodes.find(n => n.id === data.best_node_id)
                    if (!bestNode) return <div>No best path found</div>

                    // Reconstruct path from root to best node
                    const path = []
                    let currentNode = bestNode
                    while (currentNode) {
                      path.unshift(currentNode)
                      currentNode = data.nodes.find(n => n.id === currentNode.parent_id) || null
                    }

                    return path.map((node, index) => (
                      <div key={node.id} className="flex items-center gap-2 p-2 bg-slate-50 rounded">
                        <div className="text-sm font-mono">{node.id.slice(0, 8)}...</div>
                        <div className="text-sm">
                          {node.feature_name || `Depth ${node.depth}`}
                        </div>
                        {node.score !== undefined && (
                          <div className="text-sm font-bold text-green-600">
                            {node.score.toFixed(4)}
                          </div>
                        )}
                      </div>
                    ))
                  })()}
                </div>
              </div>
            )}
          </div>
        </TabsContent>
      </Tabs>
    )
  }

  // Handle Evolution visualization
  if (algorithm === 'evolution' && isEvolutionData(data)) {
    return <EvolutionVisualization data={data} />
  }

  // Fallback for mismatched algorithm/data
  return (
    <div className="flex items-center justify-center h-64 text-slate-500">
      <div className="text-center">
        <div className="text-2xl mb-2">⚠️</div>
        <p>Data format mismatch</p>
        <p className="text-sm">
          Expected {algorithm} data but received different format
        </p>
      </div>
    </div>
  )
} 