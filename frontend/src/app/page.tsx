'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Activity, Brain, Dna, Play, Square, TrendingUp, Zap, GitBranch, AlertCircle } from 'lucide-react'
import { UnifiedVisualization } from '@/components/visualization/UnifiedVisualization'
import { GenerationChart } from '@/components/generation-chart'
import { ActionRewardsChart } from '@/components/action-rewards-chart'
import { ExperimentConfig } from '@/components/experiment-config'
import { MCTSExperimentConfig } from '@/components/mcts-experiment-config'
import { ExperimentSelector } from '@/components/ExperimentSelector'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { useExperimentData } from '@/hooks/useExperimentData'

interface ExperimentStatus {
  status: string
  components: Record<string, boolean>
  config_loaded: boolean
  experiments_count: number
  algorithm?: string
}

export default function Home() {
  const [systemStatus, setSystemStatus] = useState<ExperimentStatus | null>(null)
  const [isExperimentRunning, setIsExperimentRunning] = useState(false)
  const [currentExperimentId, setCurrentExperimentId] = useState<string | null>(null)
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<'evolution' | 'mcts'>('evolution')

  // Use the new file-based experiment data hook
  const {
    data: visualizationData,
    experiments,
    selectedExperiment,
    isLoading: dataLoading,
    error: dataError,
    refreshData,
    selectExperiment,
    isPolling,
    setIsPolling
  } = useExperimentData()

  useEffect(() => {
    // Fetch initial system status
    fetchSystemStatus()
  }, [])

  const fetchSystemStatus = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/status')
      const data = await response.json()
      setSystemStatus(data)
      setIsExperimentRunning(data.status === 'running')
      if (data.algorithm) {
        setSelectedAlgorithm(data.algorithm)
      }
    } catch (error) {
      console.error('Failed to fetch system status:', error)
    }
  }

  const startExperiment = async (config: any) => {
    try {
      const endpoint = config.algorithm === 'mcts' 
        ? 'http://localhost:8000/api/mcts/start'
        : 'http://localhost:8000/api/experiments/start'
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          experiment_name: config.experimentName || 'VULCAN Experiment',
          config_overrides: config
        })
      })
      const data = await response.json()
      
      if (data.status === 'success') {
        setCurrentExperimentId(data.data.experiment_id)
        setIsExperimentRunning(true)
        setSelectedAlgorithm(config.algorithm)
        console.log('Experiment started:', data.data.experiment_id)
        
        // Start polling for updates and refresh experiment list
        setIsPolling(true)
        setTimeout(() => refreshData(), 1000) // Give backend time to create files
      } else {
        console.error('Failed to start experiment:', data.message)
      }
    } catch (error) {
      console.error('Failed to start experiment:', error)
    }
  }

  const stopExperiment = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/experiments/stop', {
        method: 'POST'
      })
      const data = await response.json()
      
      if (data.status === 'success') {
        setIsExperimentRunning(false)
        setCurrentExperimentId(null)
        setIsPolling(false)
        console.log('Experiment stopped')
        
        // Refresh data to get final state
        setTimeout(() => refreshData(), 1000)
      }
    } catch (error) {
      console.error('Failed to stop experiment:', error)
    }
  }

  // Derive algorithm from selected experiment or visualization data
  const currentAlgorithm = (() => {
    if (selectedExperiment) {
      const expData = experiments.find(exp => exp.experiment_name === selectedExperiment)
      return expData?.algorithm === 'mcts' ? 'mcts' : 'evolution'
    }
    
    if (visualizationData && 'nodes' in visualizationData && 'edges' in visualizationData) {
      return 'mcts'
    }
    
    return 'evolution'
  })() as 'evolution' | 'mcts'

  const currentGeneration = visualizationData?.generation_history?.length || 0
  const bestScore = visualizationData?.best_candidate?.score || 0
  const totalFeatures = visualizationData?.generation_history?.reduce((sum, gen) => sum + gen.total_features, 0) || 0
  const successRate = visualizationData?.generation_history?.length 
    ? visualizationData.generation_history.reduce((sum, gen) => sum + (gen.successful_features / gen.total_features), 0) / visualizationData.generation_history.length 
    : 0

  const algorithmIcon = currentAlgorithm === 'mcts' ? <GitBranch className="h-8 w-8 text-purple-600" /> : <Dna className="h-8 w-8 text-blue-600" />
  const algorithmName = currentAlgorithm === 'mcts' ? 'MCTS' : 'Evolution'

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="mb-8 text-center">
          <div className="flex items-center justify-center gap-3 mb-4">
            {algorithmIcon}
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              VULCAN 2.0
            </h1>
          </div>
          <p className="text-lg text-slate-600 dark:text-slate-400">
            Autonomous Feature Engineering with {algorithmName}
          </p>
        </div>

        {/* System Status */}
        <Card className="mb-8 border-l-4 border-l-blue-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Status
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="text-center">
                <Badge variant={systemStatus?.status === 'running' ? 'default' : 'secondary'}>
                  {systemStatus?.status || 'unknown'}
                </Badge>
                <p className="text-sm text-slate-600 mt-1">Status</p>
              </div>
              <div className="text-center">
                <Badge variant={isPolling ? 'default' : 'secondary'}>
                  {isPolling ? 'Live Updates' : 'Static View'}
                </Badge>
                <p className="text-sm text-slate-600 mt-1">Data Mode</p>
              </div>
              <div className="text-center">
                <span className="text-2xl font-bold">{experiments.length}</span>
                <p className="text-sm text-slate-600">Experiments</p>
              </div>
              <div className="text-center">
                <Badge variant={isExperimentRunning ? 'default' : 'secondary'}>
                  {isExperimentRunning ? 'Running' : 'Idle'}
                </Badge>
                <p className="text-sm text-slate-600 mt-1">Experiment</p>
              </div>
              <div className="text-center">
                <Badge variant={currentAlgorithm === 'mcts' ? 'secondary' : 'default'}>
                  {currentAlgorithm.toUpperCase()}
                </Badge>
                <p className="text-sm text-slate-600 mt-1">Algorithm</p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Data Error Display */}
        {dataError && (
          <Card className="mb-8 border-l-4 border-l-red-500">
            <CardContent className="pt-6">
              <div className="flex items-center gap-2 text-red-600">
                <AlertCircle className="h-5 w-5" />
                <span className="font-medium">Data Loading Error:</span>
                <span>{dataError}</span>
              </div>
            </CardContent>
          </Card>
        )}

        {/* Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">
                {currentAlgorithm === 'mcts' ? 'Iterations' : 'Current Generation'}
              </CardTitle>
              <Brain className="h-4 w-4 text-blue-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-blue-600">{currentGeneration}</div>
              <p className="text-xs text-slate-600">
                {currentAlgorithm === 'mcts' ? 'MCTS expansions' : 'Evolution cycles'}
              </p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Best Score</CardTitle>
              <TrendingUp className="h-4 w-4 text-green-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-green-600">{bestScore.toFixed(4)}</div>
              <p className="text-xs text-slate-600">Performance metric</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Features Generated</CardTitle>
              <Zap className="h-4 w-4 text-purple-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-purple-600">{totalFeatures}</div>
              <p className="text-xs text-slate-600">Total candidates</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
              <CardTitle className="text-sm font-medium">Success Rate</CardTitle>
              <Activity className="h-4 w-4 text-orange-600" />
            </CardHeader>
            <CardContent>
              <div className="text-2xl font-bold text-orange-600">{(successRate * 100).toFixed(1)}%</div>
              <p className="text-xs text-slate-600">Execution success</p>
              <Progress value={successRate * 100} className="mt-2" />
            </CardContent>
          </Card>
        </div>

        {/* Main Dashboard */}
        <Tabs defaultValue="results" className="space-y-6">
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="results">Results</TabsTrigger>
            <TabsTrigger value="experiment">New Experiment</TabsTrigger>
            <TabsTrigger value="visualization">
              {currentAlgorithm === 'mcts' ? 'MCTS Tree' : 'Evolution Tree'}
            </TabsTrigger>
            <TabsTrigger value="history">
              {currentAlgorithm === 'mcts' ? 'Progress History' : 'Generation History'}
            </TabsTrigger>
            <TabsTrigger value="actions">
              {currentAlgorithm === 'mcts' ? 'Node Actions' : 'RL Actions'}
            </TabsTrigger>
          </TabsList>

          <TabsContent value="results" className="space-y-6">
            <ExperimentSelector
              experiments={experiments}
              selectedExperiment={selectedExperiment}
              onSelectExperiment={selectExperiment}
              isPolling={isPolling}
              onTogglePolling={setIsPolling}
              onRefresh={refreshData}
              isLoading={dataLoading}
            />
            
            {visualizationData?.best_candidate && (
              <Card className="border-l-4 border-l-green-500">
                <CardContent className="pt-6">
                  <div className="text-center">
                    <h3 className="text-lg font-semibold text-green-800 mb-4">
                      🏆 Best Feature Candidate
                    </h3>
                    <div className="grid grid-cols-3 gap-4 text-sm">
                      <div>
                        <span className="font-medium">Name:</span> {visualizationData.best_candidate.feature_name}
                      </div>
                      <div>
                        <span className="font-medium">Score:</span> {visualizationData.best_candidate.score.toFixed(6)}
                      </div>
                      <div>
                        <span className="font-medium">
                          {currentAlgorithm === 'mcts' ? 'Depth:' : 'Generation:'}
                        </span> {visualizationData.best_candidate.generation}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>

          <TabsContent value="experiment" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Experiment Control</CardTitle>
                <CardDescription>
                  Configure and manage feature engineering experiments
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-6">
                {/* Algorithm Selection - only show when not running */}
                {!isExperimentRunning && (
                  <>
                    <div className="flex items-center gap-4">
                      <Label className="text-base">Algorithm:</Label>
                      <Select 
                        value={selectedAlgorithm} 
                        onValueChange={(value: 'evolution' | 'mcts') => setSelectedAlgorithm(value)}
                        disabled={isExperimentRunning}
                      >
                        <SelectTrigger className="w-64">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="evolution">
                            <div className="flex items-center gap-2">
                              <Dna className="h-4 w-4" />
                              Progressive Evolution
                            </div>
                          </SelectItem>
                          <SelectItem value="mcts">
                            <div className="flex items-center gap-2">
                              <GitBranch className="h-4 w-4" />
                              Monte Carlo Tree Search (MCTS)
                            </div>
                          </SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <Separator />
                  </>
                )}
                
                {/* Algorithm-specific configuration */}
                {selectedAlgorithm === 'mcts' ? (
                  <MCTSExperimentConfig 
                    onStartExperiment={startExperiment}
                    isRunning={isExperimentRunning}
                  />
                ) : (
                  <ExperimentConfig 
                    onStartExperiment={startExperiment}
                    isRunning={isExperimentRunning}
                  />
                )}
                
                <Separator />
                
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-lg font-semibold">Experiment Control</h3>
                    <p className="text-sm text-slate-600">
                      {isExperimentRunning 
                        ? `Running ${selectedAlgorithm} experiment: ${currentExperimentId?.slice(0, 8)}...`
                        : 'No experiment currently running'
                      }
                    </p>
                  </div>
                  
                  {isExperimentRunning ? (
                    <Button 
                      onClick={stopExperiment}
                      variant="destructive"
                      className="gap-2"
                    >
                      <Square className="h-4 w-4" />
                      Stop Experiment
                    </Button>
                  ) : (
                    <Button 
                      onClick={() => startExperiment({
                        experimentName: `Quick ${selectedAlgorithm === 'mcts' ? 'MCTS' : 'Evolution'} Test`,
                        algorithm: selectedAlgorithm,
                        max_iterations: selectedAlgorithm === 'mcts' ? 5 : 10,
                        data_sample_size: selectedAlgorithm === 'mcts' ? 10000 : 1000,
                        val_sample_size: selectedAlgorithm === 'mcts' ? 5000 : undefined
                      })}
                      className="gap-2"
                    >
                      <Play className="h-4 w-4" />
                      Quick Start
                    </Button>
                  )}
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="visualization" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>
                  {currentAlgorithm === 'mcts' ? 'MCTS Tree Visualization' : 'Progressive Evolution Visualization'}
                </CardTitle>
                <CardDescription>
                  File-based visualization of the feature {currentAlgorithm === 'mcts' ? 'search tree' : 'population'} and {currentAlgorithm === 'mcts' ? 'exploration' : 'evolutionary'} process
                </CardDescription>
              </CardHeader>
              <CardContent>
                <UnifiedVisualization 
                  data={visualizationData} 
                  algorithm={currentAlgorithm}
                />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="history" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>
                  {currentAlgorithm === 'mcts' ? 'Search Progress History' : 'Generation Performance History'}
                </CardTitle>
                <CardDescription>
                  Track performance improvements across {currentAlgorithm === 'mcts' ? 'iterations' : 'generations'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <GenerationChart data={visualizationData?.generation_history || []} />
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="actions" className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>
                  {currentAlgorithm === 'mcts' ? 'Node Action Distribution' : 'RL Action Rewards'}
                </CardTitle>
                <CardDescription>
                  {currentAlgorithm === 'mcts' 
                    ? 'Distribution of actions taken at each tree node'
                    : 'Reinforcement learning rewards for different actions over time'
                  }
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ActionRewardsChart 
                  data={visualizationData?.action_rewards || { generate_new: [], mutate_existing: [] }} 
                />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  )
}
