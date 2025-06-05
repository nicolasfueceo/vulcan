'use client'

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Progress } from '@/components/ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Activity, Brain, Dna, Square, TrendingUp, Zap, GitBranch, AlertCircle, Terminal, ListChecks } from 'lucide-react'
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
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { FitnessProgressionPlot } from '@/components/fitness-progression-plot'
import { ActionPerformancePlot } from '@/components/action-performance-plot'
import { ScoreDistributionPlot } from '@/components/score-distribution-plot'
import { Separator } from '@/components/ui/separator'

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
  
  const [allExperimentsList, setAllExperimentsList] = useState<UIExperiment[]>([])
  const [isLoadingAllExperiments, setIsLoadingAllExperiments] = useState<boolean>(true)
  const [allExperimentsError, setAllExperimentsError] = useState<string | null>(null)
  
  const {
    data: visualizationData,
    selectedExperiment,
    setSelectedExperiment,
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
        setQueuedExperiments(data.queued_experiments || [])
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
      fetchAllExperimentsList()
      if (isPolling) {
        refreshData()
      }
    }, 5000)

    fetchSystemStatus()
    fetchAllExperimentsList()

    return () => clearInterval(interval)
  }, [isPolling, refreshData])

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
    return 'evolution'
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
            Autonomous Feature Engineering Dashboard
          </p>
        </div>
        
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <Card className="lg:col-span-3">
            <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Activity className="h-5 w-5" />
                  System Status
                </CardTitle>
            </CardHeader>
            <CardContent>
              {systemStatus && (
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Overall Status</span>
                    <Badge variant={systemStatus.status === 'running' ? 'default' : 'destructive'}>
                      {systemStatus.status}
                    </Badge>
                  </div>
                  <Progress value={systemStatus.status === 'running' ? 100 : 0} className="w-full" />
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <div className="mt-8">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Terminal className="h-5 w-5" />
                  Experiment Queue
                </CardTitle>
                <CardDescription>These experiments are waiting to be executed.</CardDescription>
              </CardHeader>
              <CardContent className="h-48 overflow-y-auto bg-slate-100 dark:bg-slate-800 p-2 rounded-md font-mono text-xs">
                {queuedExperiments.length > 0 ? (
                  queuedExperiments.map(exp => <p key={exp.id}>[QUEUED] {exp.name} (ID: {exp.id.slice(0,8)}...)</p>)
                ) : (
                  <p>[INFO] Experiment queue is empty.</p>
                )}
              </CardContent>
            </Card>
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
              <Tabs 
                defaultValue={allExperimentsList[0]?.id} 
                onValueChange={(expId) => setSelectedExperiment(expId)}
                orientation="vertical"
              >
                <TabsList>
                  {allExperimentsList.map(exp => (
                    <TabsTrigger key={exp.id} value={exp.id}>{exp.name}</TabsTrigger>
                  ))}
                </TabsList>
                {allExperimentsList.map(exp => (
                  <TabsContent key={exp.id} value={exp.id}>
                    <div className="p-4 border rounded-md w-full">
                      <div className="flex justify-between items-center">
                        <h3 className="font-bold text-lg">{exp.name}</h3>
                        {exp.has_tensorboard_logs && (
                          <Dialog>
                            <DialogTrigger asChild>
                              <Button variant="outline">View TensorBoard Command</Button>
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader>
                                <DialogTitle>Run TensorBoard</DialogTitle>
                                <DialogDescription>
                                  To view the TensorBoard logs for this experiment, run the following command in your terminal from the project root directory.
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
                      <p>Status: <Badge variant={exp.metadata.status === 'completed' ? 'default' : 'secondary'}>{exp.metadata.status}</Badge></p>
                      <p>Started: {new Date(exp.metadata.start_time || "").toLocaleString()}</p>
                      
                      <Separator className="my-4" />

                      {visualizationData && selectedExperiment === exp.id ? (
                        <div className="space-y-4">
                          <FitnessProgressionPlot data={visualizationData.generation_history} />
                          <ActionPerformancePlot data={visualizationData.action_rewards} />
                          <ScoreDistributionPlot data={visualizationData.population} />
                        </div>
                      ) : (
                        <p>Loading visualization data...</p>
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
