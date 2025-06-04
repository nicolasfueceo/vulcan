'use client'

import { Badge } from '@/components/ui/badge'
import { Card } from '@/components/ui/card'

interface EvolutionVisualizationProps {
  data: {
    population: Array<{
      id: string
      score: number
      generation: number
      parent_id?: string
      mutation_type?: string
      execution_successful: boolean
      repair_attempts: number
    }>
    generation_history: Array<{
      generation: number
      total_features: number
      successful_features: number
      avg_score: number
      best_score: number
      action_taken: string
      population_size: number
    }>
    action_rewards: {
      generate_new: number[]
      mutate_existing: number[]
    }
    best_candidate?: {
      feature_name: string
      score: number
      generation: number
    }
  } | null
}

export function EvolutionVisualization({ data }: EvolutionVisualizationProps) {
  if (!data || !data.population.length) {
    return (
      <div className="flex items-center justify-center h-64 text-slate-500">
        <div className="text-center">
          <div className="text-2xl mb-2">🧬</div>
          <p>No evolution data available</p>
          <p className="text-sm">Start an experiment to see the population evolve</p>
        </div>
      </div>
    )
  }

  // Group population by generation
  const generationGroups = data.population.reduce((acc, individual) => {
    if (!acc[individual.generation]) {
      acc[individual.generation] = []
    }
    acc[individual.generation].push(individual)
    return acc
  }, {} as Record<number, typeof data.population>)

  const generations = Object.keys(generationGroups)
    .map(Number)
    .sort((a, b) => a - b)

  const getScoreColor = (score: number) => {
    if (score > 0.8) return 'bg-green-500'
    if (score > 0.6) return 'bg-yellow-500'
    if (score > 0.4) return 'bg-orange-500'
    return 'bg-red-500'
  }

  const getStatusBadge = (individual: any) => {
    if (!individual.execution_successful) {
      return <Badge variant="destructive" className="text-xs">Failed</Badge>
    }
    if (individual.repair_attempts > 0) {
      return <Badge variant="outline" className="text-xs">Repaired</Badge>
    }
    if (individual.mutation_type) {
      return <Badge variant="secondary" className="text-xs">{individual.mutation_type}</Badge>
    }
    return <Badge variant="default" className="text-xs">New</Badge>
  }

  return (
    <div className="space-y-6">
      {/* Population Overview */}
      <div className="grid grid-cols-3 gap-4 text-center">
        <div>
          <div className="text-2xl font-bold">{data.population.length}</div>
          <div className="text-sm text-slate-600">Population Size</div>
        </div>
        <div>
          <div className="text-2xl font-bold">{generations.length}</div>
          <div className="text-sm text-slate-600">Generations</div>
        </div>
        <div>
          <div className="text-2xl font-bold">
            {data.population.filter(p => p.execution_successful).length}
          </div>
          <div className="text-sm text-slate-600">Successful Features</div>
        </div>
      </div>

      {/* Generation Tree */}
      <div className="space-y-4 max-h-96 overflow-y-auto">
        {generations.map(generation => (
          <Card key={generation} className="p-4">
            <div className="flex items-center gap-2 mb-3">
              <Badge variant="outline">Generation {generation}</Badge>
              <span className="text-sm text-slate-600">
                {generationGroups[generation].length} features
              </span>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
              {generationGroups[generation]
                .sort((a, b) => b.score - a.score)
                .map(individual => (
                  <div
                    key={individual.id}
                    className="p-3 border rounded-lg hover:shadow-sm transition-shadow"
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-mono text-sm truncate">
                        {individual.id.slice(0, 12)}...
                      </span>
                      {getStatusBadge(individual)}
                    </div>
                    
                    <div className="flex items-center gap-2 mb-2">
                      <div 
                        className={`w-3 h-3 rounded-full ${getScoreColor(individual.score)}`}
                        title={`Score: ${individual.score.toFixed(4)}`}
                      />
                      <span className="text-sm font-medium">
                        {individual.score.toFixed(4)}
                      </span>
                    </div>

                    {individual.parent_id && (
                      <div className="text-xs text-slate-500 flex items-center gap-1">
                        <span>Parent:</span>
                        <span className="font-mono">
                          {individual.parent_id.slice(0, 8)}...
                        </span>
                      </div>
                    )}

                    {individual.repair_attempts > 0 && (
                      <div className="text-xs text-orange-600 mt-1">
                        Repaired {individual.repair_attempts}x
                      </div>
                    )}
                  </div>
                ))}
            </div>
          </Card>
        ))}
      </div>

      {/* Best Candidate Highlight */}
      {data.best_candidate && (
        <Card className="p-4 bg-gradient-to-r from-green-50 to-green-100 dark:from-green-900/20 dark:to-green-800/20 border-green-200 dark:border-green-800">
          <div className="text-center">
            <div className="text-lg font-semibold text-green-800 dark:text-green-200 mb-2">
              🏆 Current Best Feature
            </div>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <div className="text-sm text-green-600 dark:text-green-400">Name</div>
                <div className="font-mono">{data.best_candidate.feature_name}</div>
              </div>
              <div>
                <div className="text-sm text-green-600 dark:text-green-400">Score</div>
                <div className="text-xl font-bold">{data.best_candidate.score.toFixed(6)}</div>
              </div>
              <div>
                <div className="text-sm text-green-600 dark:text-green-400">Generation</div>
                <div className="text-xl font-bold">{data.best_candidate.generation}</div>
              </div>
            </div>
          </div>
        </Card>
      )}
    </div>
  )
} 