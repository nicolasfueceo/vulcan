'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Separator } from '@/components/ui/separator'
import { Activity, Brain, Dna, Play, Square, TrendingUp, Zap, GitBranch, AlertCircle, Terminal, ListChecks } from 'lucide-react'
import { ExperimentConfig } from '@/components/experiment-config'
import { MCTSExperimentConfig } from '@/components/mcts-experiment-config'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Label } from '@/components/ui/label'
import { useExperimentData } from '@/hooks/useExperimentData'
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"

interface ExperimentStatus {
  status: string
  components: Record<string, boolean>
  config_loaded: boolean
  experiments_count: number
  algorithm?: string
  queued_experiments: QueuedExperiment[]
}

interface UIMetadata {
  id: string;
  name: string;
  start_time: string | null;
  status: string;
}

interface UIExperiment {
  id: string;
  name: string;
  has_tensorboard_logs: boolean;
  tensorboard_log_path: string | null;
  metadata: UIMetadata;
}

interface UIExperimentApiResponse {
  experiments: UIExperiment[];
}

interface QueuedExperiment {
  id: string;
  name: string;
  queued_at: string;
}

interface ExperimentListItem {
  id: number | string
  experiment_name: string
  algorithm: 'evolution' | 'mcts'
}

export default function Home() {
  const [systemStatus, setSystemStatus] = useState<ExperimentStatus | null>(null)
  const [isExperimentRunning, setIsExperimentRunning] = useState(false)
  const [currentExperimentId, setCurrentExperimentId] = useState<string | null>(null)
  const [selectedAlgorithm, setSelectedAlgorithm] = useState<'evolution' | 'mcts'>('evolution')
  
  const [allExperimentsList, setAllExperimentsList] = useState<UIExperiment[]>([])
  const [isLoadingAllExperiments, setIsLoadingAllExperiments] = useState<boolean>(true)
  const [allExperimentsError, setAllExperimentsError] = useState<string | null>(null)
  
  const {
    data: visualizationData,
    selectedExperiment,
    error: dataError,
    refreshData,
    isPolling,
    setIsPolling,
    experiments: experimentsForSelectorFromHook,
  } = useExperimentData()

  const [queuedExperiments, setQueuedExperiments] = useState<QueuedExperiment[]>([])

  useEffect(() => {
    const fetchSystemStatus = async () => {
      try {
        const response = await fetch('http://localhost:8000/api/status')
        const data = await response.json()
        setSystemStatus(data)
        setIsExperimentRunning(data.status === 'running')
        setQueuedExperiments(data.queued_experiments || [])
        if (data.algorithm) {
          setSelectedAlgorithm(data.algorithm)
        }
      } catch (error) {
        console.error('Failed to fetch system status:', error)
      }
    }

    const fetchAllExperimentsList = async () => {
      setIsLoadingAllExperiments(true)
      setAllExperimentsError(null)
      try {
        const response = await fetch("http://localhost:8000/api/experiments/list")
        if (!response.ok) {
          const errorData = await response.json()
          throw new Error(errorData.message || `HTTP error! status: ${response.status}`)
        }
        const data: UIExperimentApiResponse = await response.json()
        setAllExperimentsList(data.experiments)
      } catch (e: any) {
        console.error("Failed to fetch all experiments list:", e)
        setAllExperimentsError(e.message || "Failed to load experiment list.")
      } finally {
        setIsLoadingAllExperiments(false)
      }
    }

    const interval = setInterval(() => {
      fetchSystemStatus()
    }, 5000)

    fetchSystemStatus()
    fetchAllExperimentsList()

    return () => clearInterval(interval)
  }, [])

  const startExperiment = async (config: any) => {
    try {
      const endpoint = config.algorithm === 'mcts' 
        ? 'http://localhost:8000/api/mcts/start'
        : 'http://localhost:8000/api/experiments/start'
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
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
        setIsPolling(true)
        setTimeout(() => {
          refreshData()
        }, 1000)
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
        setTimeout(() => {
          refreshData()
        }, 1000)
      } else {
        console.error('Failed to stop experiment:', data.message)
      }
    } catch (error) {
      console.error('Failed to stop experiment:', error)
    }
  }

  const derivedCurrentAlgorithm = (() => {
    if (selectedExperiment && experimentsForSelectorFromHook) {
      const expData = (experimentsForSelectorFromHook as ExperimentListItem[]).find(
        (exp: ExperimentListItem) => exp && (exp.experiment_name === selectedExperiment || String(exp.id) === selectedExperiment)
      )
      if (expData) {
        return expData.algorithm
      }
    }
    if (visualizationData && 'nodes' in visualizationData && 'edges' in visualizationData && !('generation_history' in visualizationData)) {
      return 'mcts'
    }
    return selectedAlgorithm
  })() as 'evolution' | 'mcts'

  const currentGeneration = visualizationData?.generation_history?.length || 0
  const bestScore = visualizationData?.stats?.best_score || 0
  const totalFeatures = visualizationData?.stats?.total_nodes || 0
  const successRate = visualizationData?.stats?.total_nodes && visualizationData.stats.total_nodes > 0
    ? (visualizationData.stats.total_nodes - (visualizationData.stats.failed_nodes || 0)) / visualizationData.stats.total_nodes
    : 0

  const algorithmIcon = derivedCurrentAlgorithm === 'mcts' ? <GitBranch className="h-8 w-8 text-purple-600" /> : <Dna className="h-8 w-8 text-blue-600" />
  const algorithmName = derivedCurrentAlgorithm === 'mcts' ? 'MCTS' : 'Evolution'

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text).then(() => {
      console.log("Copied to clipboard:", text)
    }).catch(err => {
      console.error("Failed to copy to clipboard:", err)
    })
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
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

        <Card className="mb-8 border-l-4 border-l-blue-500">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5" />
              System Control
            </CardTitle>
            <CardDescription>
              Start, stop, and monitor VULCAN experiments. Current algorithm: <strong>{algorithmName}</strong>
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <Label htmlFor="algorithm-selector">Select Algorithm</Label>
                <Select
                  value={selectedAlgorithm}
                  onValueChange={(value) => setSelectedAlgorithm(value as 'evolution' | 'mcts')}
                  disabled={isExperimentRunning}
                >
                  <SelectTrigger id="algorithm-selector">
                    <SelectValue placeholder="Select algorithm" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="evolution">Evolutionary Strategy</SelectItem>
                    <SelectItem value="mcts">Monte Carlo Tree Search (MCTS)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              
              <div className="flex items-end">
                {selectedAlgorithm === 'evolution' ? (
                  <ExperimentConfig onStartExperiment={startExperiment} isRunning={isExperimentRunning} onStop={stopExperiment} />
                ) : (
                  <MCTSExperimentConfig onStartExperiment={startExperiment} isRunning={isExperimentRunning} onStop={stopExperiment} />
                )}
              </div>
            </div>
            
            {systemStatus && (
              <div className="mt-6">
                <div className="flex justify-between items-center mb-2">
                  <span className="text-sm font-medium text-slate-600 dark:text-slate-400">System Status</span>
                  <Badge variant={systemStatus.status === 'running' ? 'default' : 'destructive'}>
                    {systemStatus.status}
                  </Badge>
                </div>
                <Progress value={systemStatus.status === 'running' ? 100 : 0} className="w-full" />
              </div>
            )}
            
            {dataError && (
              <Alert variant="destructive" className="mt-4">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error</AlertTitle>
                <AlertDescription>{dataError}</AlertDescription>
              </Alert>
            )}

            {currentExperimentId && (
              <div className="mt-4 p-3 bg-slate-100 dark:bg-slate-800 rounded-md">
                <p className="text-sm">
                  <strong>Current Experiment:</strong> {currentExperimentId}
                  <Button variant="ghost" size="icon" className="ml-2 h-6 w-6" onClick={() => copyToClipboard(currentExperimentId)}>
                    <Square className="h-4 w-4" />
                  </Button>
                </p>
              </div>
            )}
          </CardContent>
        </Card>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <Card className="lg:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Zap className="h-5 w-5" />
                Live Experiment View
              </CardTitle>
              <div className="flex justify-between items-center">
                <CardDescription>
                  Real-time visualization of the {algorithmName} process.
                </CardDescription>
                <div className="flex items-center gap-2">
                  <Label htmlFor="experiment-selector-main">Select Experiment</Label>
                  <Select
                    onValueChange={(value) => {
                      // Find the experiment name from the ID if needed
                      const experiment = allExperimentsList.find(e => String(e.id) === value)
                      if (experiment) {
                        setIsPolling(true)
                      }
                    }}
                    value={selectedExperiment || ""}
                  >
                    <SelectTrigger id="experiment-selector-main" className="w-[250px]">
                      <SelectValue placeholder="Select an experiment" />
                    </SelectTrigger>
                    <SelectContent>
                      {allExperimentsList.map(exp => (
                        <SelectItem key={exp.id} value={String(exp.id)}>{exp.name}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button onClick={() => setIsPolling(!isPolling)} variant={isPolling ? "secondary" : "default"}>
                    {isPolling ? 'Pause' : 'Resume'} Polling
                  </Button>
                </div>
              </div>
            </CardHeader>
            <CardContent className="h-[600px] bg-slate-50 dark:bg-slate-900 rounded-md">
              {/* This is where the main visualization will go */}
              <p className="text-center text-slate-500 dark:text-slate-400 pt-10">Select an experiment to view visualization.</p>
            </CardContent>
          </Card>

          <div className="space-y-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <TrendingUp className="h-5 w-5" />
                  Key Metrics
                </CardTitle>
                <CardDescription>High-level performance indicators.</CardDescription>
              </CardHeader>
              <CardContent className="grid grid-cols-2 gap-4">
                <div className="text-center p-4 rounded-lg bg-slate-100 dark:bg-slate-800">
                  <p className="text-sm text-slate-500 dark:text-slate-400">Best Score</p>
                  <p className="text-2xl font-bold">{bestScore.toFixed(4)}</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-slate-100 dark:bg-slate-800">
                  <p className="text-sm text-slate-500 dark:text-slate-400">Total Features</p>
                  <p className="text-2xl font-bold">{totalFeatures}</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-slate-100 dark:bg-slate-800">
                  <p className="text-sm text-slate-500 dark:text-slate-400">Current Gen</p>
                  <p className="text-2xl font-bold">{currentGeneration}</p>
                </div>
                <div className="text-center p-4 rounded-lg bg-slate-100 dark:bg-slate-800">
                  <p className="text-sm text-slate-500 dark:text-slate-400">Success Rate</p>
                  <p className="text-2xl font-bold">{(successRate * 100).toFixed(1)}%</p>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Brain className="h-5 w-5" />
                  LLM Insights
                </CardTitle>
                <CardDescription>Peek into the agent's reasoning.</CardDescription>
              </CardHeader>
              <CardContent>
                {/* To be implemented */}
                <p className="text-sm text-center text-slate-500 dark:text-slate-400">LLM thought process will appear here.</p>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Terminal className="h-5 w-5" />
                  Logs & Events
                </CardTitle>
              </CardHeader>
              <CardContent className="h-48 overflow-y-auto bg-slate-900 text-slate-200 p-2 rounded-md font-mono text-xs">
                {/* To be implemented */}
                <p>[INFO] System initialized.</p>
                <p>[INFO] Waiting for experiment to start...</p>
              </CardContent>
            </Card>
          </div>
        </div>

        <Card className="mt-8">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ListChecks className="h-5 w-5" />
              Experiment History
            </CardTitle>
            <CardDescription>Review past experiments and their outcomes.</CardDescription>
          </CardHeader>
          <CardContent>
            {isLoadingAllExperiments ? (
              <p>Loading experiments...</p>
            ) : allExperimentsError ? (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertTitle>Error Loading Experiments</AlertTitle>
                <AlertDescription>{allExperimentsError}</AlertDescription>
              </Alert>
            ) : (
              <Tabs defaultValue={allExperimentsList[0]?.id}>
                <TabsList>
                  {allExperimentsList.map(exp => (
                    <TabsTrigger key={exp.id} value={exp.id}>{exp.name}</TabsTrigger>
                  ))}
                </TabsList>
                {allExperimentsList.map(exp => (
                  <TabsContent key={exp.id} value={exp.id}>
                    <div className="p-4 border rounded-md">
                      <h3 className="font-bold">{exp.name}</h3>
                      <p>Status: <Badge variant={exp.metadata.status === 'completed' ? 'default' : 'secondary'}>{exp.metadata.status}</Badge></p>
                      <p>Started: {new Date(exp.metadata.start_time || "").toLocaleString()}</p>
                      {exp.has_tensorboard_logs && (
                        <Dialog>
                          <DialogTrigger asChild>
                            <Button className="mt-2" variant="outline">View TensorBoard Command</Button>
                          </DialogTrigger>
                          <DialogContent>
                            <DialogHeader>
                              <DialogTitle>Run TensorBoard</DialogTitle>
                              <DialogDescription>
                                To view the TensorBoard logs for this experiment, run the following command in your terminal from the project's root directory.
                              </DialogDescription>
                            </DialogHeader>
                            <div className="bg-slate-900 text-white p-4 rounded-md font-mono text-sm overflow-x-auto">
                              tensorboard --logdir {exp.tensorboard_log_path}
                            </div>
                            <DialogFooter>
                              <Button onClick={() => copyToClipboard(`tensorboard --logdir ${exp.tensorboard_log_path}`)}>
                                Copy Command
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                      )}
                    </div>
                  </TabsContent>
                ))}
              </Tabs>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
