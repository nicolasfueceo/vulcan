'use client'

import { useEffect } from 'react'
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
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { ExperimentCharts } from '@/components/ExperimentCharts'
import { LLMLogsDisplay } from '@/components/LLMLogsDisplay'

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
  const {
    experiments,
    selectedExperiment,
    setSelectedExperiment,
    experimentData,
    llmLogs,
    isLoading,
    error,
    refreshData,
  } = useExperimentData();

  useEffect(() => {
    // Initial fetch and polling setup
    const interval = setInterval(() => {
        refreshData();
    }, 5000);
    return () => clearInterval(interval);
  }, [refreshData]);

  const handleStartExperiment = async () => {
    try {
      await fetch('http://localhost:8000/api/experiments/run/preset/goodreads_large', { method: 'POST' });
    } catch (e) {
      console.error('Failed to start experiment:', e);
    }
  };

  const bestScore = experimentData?.best_candidate?.score ?? 0;
  const totalGenerations = experimentData?.generation_history?.length ?? 0;
  const populationSize = experimentData?.population?.length ?? 0;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-900 dark:to-slate-800">
      <div className="container mx-auto px-4 py-8">
        <header className="mb-8 flex flex-col items-center justify-between gap-4 sm:flex-row">
          <div className="flex items-center gap-3">
            <Dna className="h-8 w-8 text-blue-600" />
            <h1 className="text-4xl font-bold bg-gradient-to-r from-blue-600 to-purple-600 bg-clip-text text-transparent">
              VULCAN 2.0
            </h1>
          </div>
          <div className="flex items-center gap-4">
            <Select onValueChange={setSelectedExperiment} value={selectedExperiment || ''}>
              <SelectTrigger className="w-[280px]">
                <SelectValue placeholder="Select an experiment" />
              </SelectTrigger>
              <SelectContent>
                {experiments.map(exp => (
                  <SelectItem key={exp.id} value={exp.id}>
                    {exp.name} ({exp.status})
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button onClick={handleStartExperiment}>
              <Zap className="mr-2 h-4 w-4" />
              Start New Experiment
            </Button>
          </div>
        </header>

        {isLoading && !experimentData && <p className="text-center">Loading experiment data...</p>}
        {error && (
          <Alert variant="destructive" className="my-4">
            <AlertCircle className="h-4 w-4" />
            <AlertTitle>Error</AlertTitle>
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}

        {experimentData ? (
          <Tabs defaultValue="dashboard" className="w-full">
            <TabsList className="grid w-full grid-cols-3">
              <TabsTrigger value="dashboard">
                <Activity className="mr-2 h-4 w-4" /> Dashboard
              </TabsTrigger>
              <TabsTrigger value="plots">
                <TrendingUp className="mr-2 h-4 w-4" /> Performance Plots
              </TabsTrigger>
              <TabsTrigger value="logs">
                <Brain className="mr-2 h-4 w-4" /> LLM Logs
              </TabsTrigger>
            </TabsList>
            
            <TabsContent value="dashboard">
              <Card className="mt-4">
                <CardHeader>
                  <CardTitle>Experiment Dashboard</CardTitle>
                  <CardDescription>
                    Overview of the currently selected experiment.
                  </CardDescription>
                </CardHeader>
                <CardContent className="grid gap-4 md:grid-cols-3">
                  <Card>
                    <CardHeader>
                      <CardTitle>Best Score</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-3xl font-bold">{bestScore.toFixed(4)}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Generations</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-3xl font-bold">{totalGenerations}</p>
                    </CardContent>
                  </Card>
                  <Card>
                    <CardHeader>
                      <CardTitle>Population Size</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <p className="text-3xl font-bold">{populationSize}</p>
                    </CardContent>
                  </Card>
                </CardContent>
              </Card>
            </TabsContent>

            <TabsContent value="plots">
              <div className="mt-4">
                <ExperimentCharts data={experimentData} />
              </div>
            </TabsContent>

            <TabsContent value="logs">
              <div className="mt-4">
                <LLMLogsDisplay logs={llmLogs} />
              </div>
            </TabsContent>
          </Tabs>
        ) : (
          !isLoading && (
            <Card className="mt-8 text-center">
              <CardHeader>
                <CardTitle>No Experiment Data</CardTitle>
              </CardHeader>
              <CardContent>
                <p>
                  No data to display. Start a new experiment or select one from the list.
                </p>
              </CardContent>
            </Card>
          )
        )}
      </div>
    </div>
  )
}
