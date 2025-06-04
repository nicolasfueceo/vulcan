'use client'

import React from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { RefreshCw, Play, Square, Clock, TrendingUp } from 'lucide-react'

interface ExperimentListItem {
  id: number
  experiment_name: string
  algorithm: string
  start_time: string
  status: string
  iterations_completed: number
  best_score: number
  end_time?: string
}

interface ExperimentSelectorProps {
  experiments: ExperimentListItem[]
  selectedExperiment: string | null
  onSelectExperiment: (experimentName: string | null) => void
  isPolling: boolean
  onTogglePolling: (polling: boolean) => void
  onRefresh: () => void
  isLoading: boolean
}

export function ExperimentSelector({
  experiments,
  selectedExperiment,
  onSelectExperiment,
  isPolling,
  onTogglePolling,
  onRefresh,
  isLoading
}: ExperimentSelectorProps) {
  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString()
  }

  const getStatusBadge = (status: string) => {
    const variants = {
      'running': 'default',
      'completed': 'secondary',
      'failed': 'destructive'
    } as const

    return (
      <Badge variant={variants[status as keyof typeof variants] || 'outline'}>
        {status.toUpperCase()}
      </Badge>
    )
  }

  const getAlgorithmBadge = (algorithm: string) => {
    const colors = {
      'mcts': 'bg-purple-100 text-purple-800',
      'evolution': 'bg-blue-100 text-blue-800'
    }

    return (
      <Badge className={`${colors[algorithm as keyof typeof colors] || 'bg-gray-100 text-gray-800'} text-xs`}>
        {algorithm.toUpperCase()}
      </Badge>
    )
  }

  const selectedExpData = experiments.find(exp => exp.experiment_name === selectedExperiment)

  return (
    <Card>
      <CardHeader>
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2">
            <Clock className="h-5 w-5" />
            Experiment Results
          </CardTitle>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={onRefresh}
              disabled={isLoading}
            >
              <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
            <Button
              variant={isPolling ? "destructive" : "default"}
              size="sm"
              onClick={() => onTogglePolling(!isPolling)}
              disabled={!selectedExpData || selectedExpData.status !== 'running'}
            >
              {isPolling ? (
                <>
                  <Square className="h-4 w-4 mr-1" />
                  Stop Live
                </>
              ) : (
                <>
                  <Play className="h-4 w-4 mr-1" />
                  Live Updates
                </>
              )}
            </Button>
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Experiment Selector */}
        <div className="flex items-center gap-4">
          <Label className="text-base">Select Experiment:</Label>
          <Select value={selectedExperiment || ""} onValueChange={(value) => onSelectExperiment(value || null)}>
            <SelectTrigger className="w-64">
              <SelectValue placeholder="Choose experiment..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="">Latest Experiment</SelectItem>
              {experiments.map((exp) => (
                <SelectItem key={exp.experiment_name} value={exp.experiment_name}>
                  <div className="flex items-center gap-2">
                    {getAlgorithmBadge(exp.algorithm)}
                    <span className="truncate max-w-32">
                      {exp.experiment_name}
                    </span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Current Experiment Info */}
        {selectedExpData && (
          <div className="p-4 bg-slate-50 rounded-lg space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                {getAlgorithmBadge(selectedExpData.algorithm)}
                {getStatusBadge(selectedExpData.status)}
                {isPolling && selectedExpData.status === 'running' && (
                  <Badge variant="outline" className="text-green-600 border-green-600">
                    LIVE
                  </Badge>
                )}
              </div>
              <div className="flex items-center gap-1 text-green-600">
                <TrendingUp className="h-4 w-4" />
                <span className="font-bold">{selectedExpData.best_score.toFixed(4)}</span>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="font-medium">Experiment:</span>
                <div className="font-mono text-xs">{selectedExpData.experiment_name}</div>
              </div>
              <div>
                <span className="font-medium">Iterations:</span>
                <div>{selectedExpData.iterations_completed}</div>
              </div>
              <div>
                <span className="font-medium">Started:</span>
                <div>{formatTime(selectedExpData.start_time)}</div>
              </div>
              {selectedExpData.end_time && (
                <div>
                  <span className="font-medium">Finished:</span>
                  <div>{formatTime(selectedExpData.end_time)}</div>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Experiments List */}
        {experiments.length > 0 && (
          <div className="space-y-2">
            <Label className="text-sm font-medium">Recent Experiments ({experiments.length})</Label>
            <div className="max-h-48 overflow-y-auto space-y-1">
              {experiments.slice(0, 10).map((exp) => (
                <div
                  key={exp.experiment_name}
                  className={`p-2 border rounded cursor-pointer hover:bg-slate-50 transition-colors ${
                    exp.experiment_name === selectedExperiment ? 'border-blue-500 bg-blue-50' : 'border-slate-200'
                  }`}
                  onClick={() => onSelectExperiment(exp.experiment_name)}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      {getAlgorithmBadge(exp.algorithm)}
                      {getStatusBadge(exp.status)}
                      <span className="text-sm font-mono truncate max-w-32">
                        {exp.experiment_name}
                      </span>
                    </div>
                    <div className="text-sm text-slate-600">
                      Score: {exp.best_score.toFixed(4)}
                    </div>
                  </div>
                  <div className="text-xs text-slate-500 mt-1">
                    {formatTime(exp.start_time)} • {exp.iterations_completed} iterations
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {experiments.length === 0 && (
          <div className="text-center text-slate-500 py-8">
            <div className="text-2xl mb-2">📁</div>
            <p>No experiments found</p>
            <p className="text-sm">Run an experiment to see results here</p>
          </div>
        )}
      </CardContent>
    </Card>
  )
} 