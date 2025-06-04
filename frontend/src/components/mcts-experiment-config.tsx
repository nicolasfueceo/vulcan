'use client'

import { useState } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Play, Settings, Brain } from 'lucide-react'

interface MCTSConfigProps {
  onStartExperiment: (config: any) => void
  isRunning: boolean
}

export function MCTSExperimentConfig({ onStartExperiment, isRunning }: MCTSConfigProps) {
  const [config, setConfig] = useState({
    experimentName: 'MCTS Feature Engineering',
    algorithm: 'mcts',
    max_iterations: 10,
    max_depth: 5,
    exploration_factor: 1.4,
    llm_temperature: 0.7,
    llm_model: 'gpt-4o-mini',
    // Data configuration
    data_sample_size: 20000,  // Increased for better validation
    val_sample_size: 10000,   // New: explicit validation size
    test_sample_size: 5000,
    outer_fold: 1,
    inner_fold: 1,
    // Evaluation parameters
    fast_evaluation: true,
    n_components: 50,
    max_epochs: 25
  })

  const handleInputChange = (key: string, value: string | number | boolean) => {
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
      experimentName: 'Quick MCTS Test',
      algorithm: 'mcts',
      max_iterations: 5,
      max_depth: 3,
      exploration_factor: 1.4,
      llm_temperature: 0.7,
      llm_model: 'gpt-4o-mini',
      data_sample_size: 10000,
      val_sample_size: 5000,
      test_sample_size: 2000,
      outer_fold: 1,
      inner_fold: 1,
      fast_evaluation: true,
      n_components: 30,
      max_epochs: 20
    },
    standard: {
      experimentName: 'Standard MCTS Search',
      algorithm: 'mcts',
      max_iterations: 10,
      max_depth: 5,
      exploration_factor: 1.4,
      llm_temperature: 0.7,
      llm_model: 'gpt-4o-mini',
      data_sample_size: 20000,
      val_sample_size: 10000,
      test_sample_size: 5000,
      outer_fold: 1,
      inner_fold: 1,
      fast_evaluation: true,
      n_components: 50,
      max_epochs: 25
    },
    intensive: {
      experimentName: 'Intensive MCTS Search',
      algorithm: 'mcts',
      max_iterations: 20,
      max_depth: 7,
      exploration_factor: 1.0,
      llm_temperature: 0.8,
      llm_model: 'gpt-4',
      data_sample_size: 50000,
      val_sample_size: 25000,
      test_sample_size: 10000,
      outer_fold: 1,
      inner_fold: 1,
      fast_evaluation: false,
      n_components: 100,
      max_epochs: 50
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
            MCTS Quick Presets
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
              <span className="text-sm text-slate-600">5 iterations, 5K val samples</span>
              <span className="text-xs text-slate-500">~5-10 minutes</span>
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => loadPreset('standard')}
              className="p-4 h-auto flex flex-col items-start"
              disabled={isRunning}
            >
              <span className="font-semibold">⚖️ Standard</span>
              <span className="text-sm text-slate-600">10 iterations, 10K val samples</span>
              <span className="text-xs text-slate-500">~20-30 minutes</span>
            </Button>
            
            <Button 
              variant="outline" 
              onClick={() => loadPreset('intensive')}
              className="p-4 h-auto flex flex-col items-start"
              disabled={isRunning}
            >
              <span className="font-semibold">🔥 Intensive</span>
              <span className="text-sm text-slate-600">20 iterations, 25K val samples</span>
              <span className="text-xs text-slate-500">~60-90 minutes</span>
            </Button>
          </div>
        </CardContent>
      </Card>

      {/* Custom Configuration */}
      <Card>
        <CardHeader>
          <CardTitle>MCTS Configuration</CardTitle>
          <CardDescription>
            Configure Monte Carlo Tree Search parameters for feature engineering
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
                <Label htmlFor="llm_model">LLM Model</Label>
                <Select 
                  value={config.llm_model} 
                  onValueChange={(value) => handleInputChange('llm_model', value)}
                  disabled={isRunning}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="gpt-4o-mini">GPT-4o Mini (Fast)</SelectItem>
                    <SelectItem value="gpt-4">GPT-4 (Best Quality)</SelectItem>
                    <SelectItem value="gpt-3.5-turbo">GPT-3.5 Turbo (Budget)</SelectItem>
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          <Separator />

          {/* MCTS Parameters */}
          <div className="space-y-4">
            <h4 className="font-semibold flex items-center gap-2">
              <Brain className="h-4 w-4" />
              MCTS Parameters
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="max_iterations">Max Iterations</Label>
                <Input
                  id="max_iterations"
                  type="number"
                  value={config.max_iterations}
                  onChange={(e) => handleInputChange('max_iterations', e.target.value)}
                  min="1"
                  max="100"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Tree expansions</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="max_depth">Max Depth</Label>
                <Input
                  id="max_depth"
                  type="number"
                  value={config.max_depth}
                  onChange={(e) => handleInputChange('max_depth', e.target.value)}
                  min="1"
                  max="10"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Feature chain length</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="exploration_factor">Exploration Factor</Label>
                <Input
                  id="exploration_factor"
                  type="number"
                  step="0.1"
                  value={config.exploration_factor}
                  onChange={(e) => handleInputChange('exploration_factor', e.target.value)}
                  min="0.5"
                  max="2.0"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">UCB constant (c)</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="llm_temperature">LLM Temperature</Label>
                <Input
                  id="llm_temperature"
                  type="number"
                  step="0.1"
                  value={config.llm_temperature}
                  onChange={(e) => handleInputChange('llm_temperature', e.target.value)}
                  min="0.0"
                  max="1.0"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Creativity level</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Data Configuration */}
          <div className="space-y-4">
            <h4 className="font-semibold">Data Configuration</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <div className="space-y-2">
                <Label htmlFor="data_sample_size">Train Sample Size</Label>
                <Input
                  id="data_sample_size"
                  type="number"
                  value={config.data_sample_size}
                  onChange={(e) => handleInputChange('data_sample_size', e.target.value)}
                  min="1000"
                  max="100000"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Training data</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="val_sample_size">Validation Size</Label>
                <Input
                  id="val_sample_size"
                  type="number"
                  value={config.val_sample_size}
                  onChange={(e) => handleInputChange('val_sample_size', e.target.value)}
                  min="1000"
                  max="50000"
                  disabled={isRunning}
                  className="border-blue-300"
                />
                <p className="text-xs text-blue-600 font-medium">For optimization</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="test_sample_size">Test Sample Size</Label>
                <Input
                  id="test_sample_size"
                  type="number"
                  value={config.test_sample_size}
                  onChange={(e) => handleInputChange('test_sample_size', e.target.value)}
                  min="1000"
                  max="50000"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Final evaluation</p>
              </div>
              
              <div className="space-y-2">
                <Label>Data Splits</Label>
                <div className="flex gap-2">
                  <Select 
                    value={config.outer_fold.toString()} 
                    onValueChange={(value) => handleInputChange('outer_fold', value)}
                    disabled={isRunning}
                  >
                    <SelectTrigger className="w-20">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[1, 2, 3, 4, 5].map(fold => (
                        <SelectItem key={fold} value={fold.toString()}>
                          O{fold}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  
                  <Select 
                    value={config.inner_fold.toString()} 
                    onValueChange={(value) => handleInputChange('inner_fold', value)}
                    disabled={isRunning}
                  >
                    <SelectTrigger className="w-20">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      {[1, 2, 3].map(fold => (
                        <SelectItem key={fold} value={fold.toString()}>
                          I{fold}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <p className="text-xs text-slate-600">Outer/Inner fold</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Evaluation Settings */}
          <div className="space-y-4">
            <h4 className="font-semibold">Evaluation Settings</h4>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="space-y-2">
                <Label htmlFor="n_components">Model Components</Label>
                <Input
                  id="n_components"
                  type="number"
                  value={config.n_components}
                  onChange={(e) => handleInputChange('n_components', e.target.value)}
                  min="10"
                  max="200"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">LightFM dimensions</p>
              </div>
              
              <div className="space-y-2">
                <Label htmlFor="max_epochs">Training Epochs</Label>
                <Input
                  id="max_epochs"
                  type="number"
                  value={config.max_epochs}
                  onChange={(e) => handleInputChange('max_epochs', e.target.value)}
                  min="10"
                  max="100"
                  disabled={isRunning}
                />
                <p className="text-xs text-slate-600">Model training</p>
              </div>
              
              <div className="space-y-2">
                <Label>Evaluation Mode</Label>
                <Select 
                  value={config.fast_evaluation ? 'fast' : 'full'} 
                  onValueChange={(value) => handleInputChange('fast_evaluation', value === 'fast')}
                  disabled={isRunning}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="fast">Fast (Rigorous on subset)</SelectItem>
                    <SelectItem value="full">Full (Complete data)</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-slate-600">Trade-off speed vs accuracy</p>
              </div>
            </div>
          </div>

          <Separator />

          {/* Start Experiment */}
          <div className="flex justify-between items-center">
            <div>
              <p className="text-sm text-slate-600">
                Ready to start MCTS search with {config.max_iterations} iterations
              </p>
              <p className="text-xs text-slate-500">
                Using {config.val_sample_size.toLocaleString()} validation samples for optimization
              </p>
              <p className="text-xs text-slate-500">
                Estimated time: {config.max_iterations < 5 ? '5-10' : config.max_iterations < 15 ? '20-40' : '60-120'} minutes
              </p>
            </div>
            
            <Button 
              onClick={handleStartExperiment}
              disabled={isRunning}
              className="gap-2"
              size="lg"
            >
              <Play className="h-4 w-4" />
              Start MCTS Search
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
} 