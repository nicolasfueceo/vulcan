'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Play, Settings } from 'lucide-react'

interface ExperimentConfigProps {
  onStartExperiment: (config: any) => void
  isRunning: boolean
}

export function ExperimentConfig({ onStartExperiment, isRunning }: ExperimentConfigProps) {
  const [config, setConfig] = useState({
    experimentName: 'Progressive Evolution Experiment',
    max_iterations: 20,
    population_size: 30,
    generation_size: 15,
    data_sample_size: 2000,
    outer_fold: 1,
    inner_fold: 1,
    max_repair_attempts: 3,
    mutation_rate: 0.3
  })

  const handleInputChange = (key: string, value: string | number) => {
    setConfig(prev => ({
      ...prev,
      [key]: typeof value === 'string' && !isNaN(Number(value)) ? Number(value) : value
    }))
  }

  const handleStartExperiment = () => {
    onStartExperiment(config)
  }

  const presetConfigs = {
    quick: {
      experimentName: 'Quick Test',
      max_iterations: 5,
      population_size: 10,
      generation_size: 8,
      data_sample_size: 500,
      outer_fold: 1,
      inner_fold: 1,
      max_repair_attempts: 2,
      mutation_rate: 0.3
    },
    standard: {
      experimentName: 'Standard Evolution',
      max_iterations: 20,
      population_size: 30,
      generation_size: 15,
      data_sample_size: 2000,
      outer_fold: 1,
      inner_fold: 1,
      max_repair_attempts: 3,
      mutation_rate: 0.3
    },
    intensive: {
      experimentName: 'Intensive Evolution',
      max_iterations: 50,
      population_size: 50,
      generation_size: 25,
      data_sample_size: 5000,
      outer_fold: 1,
      inner_fold: 1,
      max_repair_attempts: 3,
      mutation_rate: 0.25
    }
  }

  const loadPreset = (preset: keyof typeof presetConfigs) => {
    setConfig(presetConfigs[preset])
  }

  return (
    <div className="space-y-6">
      {/* Preset Configurations */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Settings className="h-5 w-5" />
            Quick Presets
          </CardTitle>
          <CardDescription>
            Choose a preset configuration to get started quickly
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <Button 
              variant="outline" 
              onClick={() => loadPreset('quick')}
              className="p-4 h-auto flex flex-col items-start"
              disabled={isRunning}
            >
              <span className="font-semibold">🚀 Quick Test</span>
              <span className="text-sm text-slate-600">5 generations, 500 samples</span>
              <span className="text-xs text-slate-500">~2-3 minutes</span>
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => loadPreset('standard')}
              className="p-4 h-auto flex flex-col items-start"
              disabled={isRunning}
            >
              <span className="font-semibold">⚖️ Standard</span>
              <span className="text-sm text-slate-600">20 generations, 2K samples</span>
              <span className="text-xs text-slate-500">~10-15 minutes</span>
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => loadPreset('intensive')}
              className="p-4 h-auto flex flex-col items-start"
              disabled={isRunning}
            >
              <span className="font-semibold">🔥 Intensive</span>
              <span className="text-sm text-slate-600">50 generations, 5K samples</span>
              <span className="text-xs text-slate-500">~30-45 minutes</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Custom Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>Custom Configuration</CardTitle>
          <CardDescription>
            Fine-tune the evolution parameters for your specific needs
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Basic Settings */}
          <div className="space-y-4">
            <h4 className="font-semibold">Basic Settings</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="experimentName">Experiment Name</Label>
                <Input
                  id="experimentName"
                  value={config.experimentName}
                  onChange={(e) => handleInputChange('experimentName', e.target.value)}
                  placeholder="Enter experiment name"
                  disabled={isRunning}
                />
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="max_iterations">Max Generations</Label>
                <Input
                  id="max_iterations"
                  type="number"
                  value={config.max_iterations}
                  onChange={(e) => handleInputChange('max_iterations', e.target.value)}
                  min="1"
                  max="100"
                  disabled={isRunning}
                />
              </div>
            </div>
          </div>

          <Separator />

          {/* Evolution Parameters */}
          <div className="space-y-4">
            <h4 className="font-semibold">Evolution Parameters</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="population_size">Population Size</Label>
                <Input
                  id="population_size"
                  type="number"
                  value={config.population_size}
                  onChange={(e) => handleInputChange('population_size', e.target.value)}
                  min="5"
                  max="100"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Best features to keep</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="generation_size">Generation Size</Label>
                <Input
                  id="generation_size"
                  type="number"
                  value={config.generation_size}
                  onChange={(e) => handleInputChange('generation_size', e.target.value)}
                  min="3"
                  max="50"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Features per generation</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="mutation_rate">Mutation Rate</Label>
                <Input
                  id="mutation_rate"
                  type="number"
                  step="0.1"
                  value={config.mutation_rate}
                  onChange={(e) => handleInputChange('mutation_rate', e.target.value)}
                  min="0.0"
                  max="1.0"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Exploration vs exploitation</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Data Configuration */}
          <div className="space-y-4">
            <h4 className="font-semibold">Data Configuration</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="data_sample_size">Sample Size</Label>
                <Input
                  id="data_sample_size"
                  type="number"
                  value={config.data_sample_size}
                  onChange={(e) => handleInputChange('data_sample_size', e.target.value)}
                  min="100"
                  max="10000"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Training data size</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="outer_fold">Outer Fold</Label>
                <Select 
                  value={config.outer_fold.toString()} 
                  onValueChange={(value) => handleInputChange('outer_fold', value)}
                  disabled={isRunning}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[1, 2, 3, 4, 5].map(fold => (
                      <SelectItem key={fold} value={fold.toString()}>
                        Fold {fold}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="inner_fold">Inner Fold</Label>
                <Select 
                  value={config.inner_fold.toString()} 
                  onValueChange={(value) => handleInputChange('inner_fold', value)}
                  disabled={isRunning}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {[1, 2, 3].map(fold => (
                      <SelectItem key={fold} value={fold.toString()}>
                        Fold {fold}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator />

          {/* Advanced Settings */}
          <div className="space-y-4">
            <h4 className="font-semibold">Advanced Settings</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="max_repair_attempts">Max Repair Attempts</Label>
                <Input
                  id="max_repair_attempts"
                  type="number"
                  value={config.max_repair_attempts}
                  onChange={(e) => handleInputChange('max_repair_attempts', e.target.value)}
                  min="0"
                  max="5"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Auto-fix failed features</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Start Experiment */}
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm text-slate-600">
                Ready to start evolution with {config.max_iterations} generations
              </p>
              <p className="text-xs text-slate-500">
                Estimated time: {config.max_iterations < 10 ? '2-5' : config.max_iterations < 30 ? '10-20' : '30-60'} minutes
              </p>
            </div>
            
            <Button 
              onClick={handleStartExperiment}
              disabled={isRunning}
              className="gap-2"
              size="lg"
            >
              <Play className="h-4 w-4" />
              Start Evolution
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 