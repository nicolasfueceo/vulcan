'use client';

import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { LLMInteractionLog } from '@/types/vulcan';
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from '@/components/ui/accordion';
import { format } from 'date-fns';

interface LLMLogsDisplayProps {
  logs: LLMInteractionLog[];
}

function syntaxHighlight(json: object) {
  if (json == null) return '<null>';
  const jsonString = JSON.stringify(json, undefined, 2);
  return jsonString.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

export function LLMLogsDisplay({ logs }: LLMLogsDisplayProps) {
  if (!logs || logs.length === 0) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>LLM Interaction Logs</CardTitle>
        </CardHeader>
        <CardContent>
          <p>No LLM interaction logs for this experiment.</p>
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>LLM Interaction Logs ({logs.length})</CardTitle>
      </CardHeader>
      <CardContent>
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Timestamp</TableHead>
              <TableHead>Agent</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Details</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {logs.map((log, index) => (
              <TableRow key={index}>
                <TableCell>
                  {format(new Date(log.timestamp * 1000), 'yyyy-MM-dd HH:mm:ss')}
                </TableCell>
                <TableCell>{log.agent_name}</TableCell>
                <TableCell>
                  {log.error_message ? (
                    <Badge variant="destructive">Error</Badge>
                  ) : (
                    <Badge variant="secondary">Success</Badge>
                  )}
                </TableCell>
                <TableCell>
                  <Accordion type="single" collapsible className="w-full">
                    <AccordionItem value={`item-${index}`}>
                      <AccordionTrigger>View Details</AccordionTrigger>
                      <AccordionContent>
                        <div className="space-y-4">
                          <div>
                            <h4 className="font-semibold">Prompt Input</h4>
                            <pre className="mt-2 rounded-md bg-slate-950 p-4">
                              <code
                                className="text-white"
                                dangerouslySetInnerHTML={{
                                  __html: syntaxHighlight(log.prompt_input),
                                }}
                              />
                            </pre>
                          </div>
                          <div>
                            <h4 className="font-semibold">Raw Response</h4>
                            <pre className="mt-2 rounded-md bg-slate-950 p-4 text-white">
                              {log.raw_response || '<empty response>'}
                            </pre>
                          </div>
                          {log.error_message && (
                            <div>
                              <h4 className="font-semibold text-red-500">Error</h4>
                              <pre className="mt-2 rounded-md bg-slate-950 p-4 text-red-400">
                                {log.error_message}
                              </pre>
                            </div>
                          )}
                        </div>
                      </AccordionContent>
                    </AccordionItem>
                  </Accordion>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </CardContent>
    </Card>
  );
} 