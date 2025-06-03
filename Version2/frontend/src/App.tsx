import React, { useState, useEffect, FormEvent, useRef, useCallback } from 'react';
import axios, { CancelTokenSource } from 'axios';
import {
  Container, Row, Col, Form, Button, ListGroup, Spinner, Alert, Card, Nav, Tab, Table, Badge,
  Accordion, ButtonGroup
} from 'react-bootstrap';
import 'bootstrap/dist/css/bootstrap.min.css';
import Plot from 'react-plotly.js';
import ReactMarkdown from 'react-markdown';

// Define interfaces for our data structures
interface QueryResult {
  [key: string]: any;
}

interface VisualizationData {
  data: Plotly.Data[];
  layout: Partial<Plotly.Layout>;
}

interface RagResult {
  answer: string | null;
  sources: string[] | null;
  confidence: number | null;
  source_types: string[] | null;
}

interface ChatMessage {
  id: string;
  sender: 'user' | 'ai';
  text?: string;
  query_results?: QueryResult[] | null;
  sql_query?: string | null;
  analysis?: string | null;
  combined_analysis?: string | null;
  visualization?: VisualizationData | string | null;
  rag_result?: RagResult | null;
  error?: string | null;
  type: 'live' | 'cache' | 'error' | 'info';
  original_question?: string | null;
  cached_at?: string | null;
  usage_count?: number | null;
  feedback_given?: boolean;
  isLoadingResponse?: boolean;
}

// For API responses before they are mapped to ChatMessage
interface BackendResponseData {
  question?: string;
  sql_query?: string;
  query_results?: QueryResult[];
  analysis?: string;
  combined_analysis?: string;
  visualization?: VisualizationData | string | null;
  rag_result?: RagResult;
  error?: string;
  // fields for cache entries
  original_question?: string;
  cached_at?: string;
  usage_count?: number;
}

interface DbInfo {
    host?: string;
    port?: string;
    database?: string;
    username?: string;
    password_masked?: string;
    connection_type?: string;
    schema?: string; // Typically the main schema name like 'public' or 'northwind'
    raw_url?: string;
    error?: string;
}

// Corrected SchemaInfo structure
interface TableDetails {
    columns: { name: string; type: string; }[];
    // foreign_keys?: { constrained_columns: string[]; referred_table: string; referred_columns: string[] }[]; // Optional
}
interface SchemaData {
    [tableName: string]: TableDetails;
}
interface SchemaInfo {
    data?: SchemaData | null;
    error?: string | null;
}

interface CacheSummaryData {
    cache_size: number;
    cache_hits: number;
    cached_items: ChatMessage[];
    error?: string;
}

const App: React.FC = () => {
  const [userInput, setUserInput] = useState<string>('');
  const [chatHistory, setChatHistory] = useState<ChatMessage[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null); // Global error for the form
  const [dbInfo, setDbInfo] = useState<DbInfo | null>(null);
  const [schemaInfo, setSchemaInfo] = useState<SchemaInfo | null>(null); // Uses the corrected SchemaInfo type
  const [cacheSummary, setCacheSummary] = useState<CacheSummaryData | null>(null);
  const [activeTab, setActiveTab] = useState('chat');
  const chatEndRef = useRef<null | HTMLDivElement>(null);
  const currentRequestAbortController = useRef<AbortController | null>(null); // Ref for AbortController

  const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5001';

  const fetchDbInfo = useCallback(async () => {
    try {
      const response = await axios.get<DbInfo>(`${API_URL}/db-info`);
      setDbInfo(response.data);
    } catch (err) {
      console.error("Error fetching DB info:", err);
      setDbInfo({ error: "Failed to load DB info." });
    }
  }, [API_URL]);

  const fetchSchemaInfo = useCallback(async () => {
    try {
      // Expect response.data to be SchemaData directly if successful
      const response = await axios.get<SchemaData>(`${API_URL}/schema-info`);
      setSchemaInfo({ data: response.data, error: null });
    } catch (err: any) {
      console.error("Error fetching schema info:", err);
      const errorMessage = err.response?.data?.error || "Failed to load schema info.";
      setSchemaInfo({ data: null, error: errorMessage });
    }
  }, [API_URL]);

  const fetchCacheSummary = useCallback(async () => {
    try {
      const response = await axios.get<CacheSummaryData>(`${API_URL}/cache-summary`);
      setCacheSummary(response.data);
    } catch (err) {
      console.error("Error fetching cache summary:", err);
      setCacheSummary({ error: "Failed to load cache summary.", cache_size: 0, cache_hits: 0, cached_items: [] });
    }
  }, [API_URL]);

  useEffect(() => {
    fetchDbInfo();
    fetchSchemaInfo();
    fetchCacheSummary();
  }, [fetchDbInfo, fetchSchemaInfo, fetchCacheSummary]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  const executeUserQuery = async (questionToSubmit: string) => {
    if (currentRequestAbortController.current) {
      currentRequestAbortController.current.abort();
    }
    currentRequestAbortController.current = new AbortController();

    const userMessageId = Date.now().toString();
    const userMessage: ChatMessage = {
      id: userMessageId,
      sender: 'user',
      text: questionToSubmit,
      type: 'info',
      isLoadingResponse: true,
    };
    setChatHistory(prev => [...prev, userMessage]);
    setIsLoading(true); // Global loading for Cancel button visibility
    setError(null); // Clear global form error
    
    let requestError: any = null;

    try {
      const apiResponse = await axios.post<{ response: BackendResponseData; type: 'live' | 'cache' }>(
        `${API_URL}/query`, 
        { question: questionToSubmit }, 
        { signal: currentRequestAbortController.current.signal }
      );
      
      const backendData = apiResponse.data.response;
      const responseType = apiResponse.data.type;

      const aiResponseMessage: ChatMessage = {
        id: Date.now().toString() + '-ai',
        sender: 'ai',
        text: backendData.combined_analysis || backendData.analysis || undefined,
        sql_query: backendData.sql_query || null,
        query_results: backendData.query_results || null,
        analysis: backendData.analysis || null,
        combined_analysis: backendData.combined_analysis || null,
        visualization: backendData.visualization || null,
        rag_result: backendData.rag_result || null,
        error: backendData.error || null,
        type: responseType,
        original_question: questionToSubmit, 
      };
      setChatHistory(prev => [...prev, aiResponseMessage]);
    } catch (err: any) {
      requestError = err;
      if (axios.isCancel(err)) {
        console.log('Request canceled by user:', err.message);
        const cancelMessage: ChatMessage = {
          id: Date.now().toString() + '-cancel',
          sender: 'ai',
          text: "Your previous query was canceled.",
          type: 'info',
        };
        setChatHistory(prev => [...prev, cancelMessage]);
      } else {
        const errorMessage = err.response?.data?.error || "An unexpected error occurred.";
        const aiError: ChatMessage = {
          id: Date.now().toString() + '-error',
          sender: 'ai',
          error: errorMessage,
          type: 'error',
          original_question: questionToSubmit, 
        };
        setChatHistory(prev => [...prev, aiError]);
        setError(errorMessage); 
      }
    } finally {
      setIsLoading(false);
      currentRequestAbortController.current = null; 
      setChatHistory(prev => 
        prev.map(msg => 
          msg.id === userMessageId ? { ...msg, isLoadingResponse: false } : msg
        )
      );
      if (!requestError || !axios.isCancel(requestError)) {
          fetchCacheSummary();
      }
    }
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!userInput.trim()) return;
    const questionToSubmit = userInput;
    setUserInput(''); // Clear input immediately
    await executeUserQuery(questionToSubmit);
  };

  const handleCancelQuery = () => {
    if (currentRequestAbortController.current) {
      currentRequestAbortController.current.abort();
    }
  };

  const handleRetry = async (chatItem: ChatMessage) => {
    if (!chatItem.original_question) {
      console.warn("Cannot retry: original question missing.");
      // Optionally, add a silent log or a non-intrusive UI update if needed
      return;
    }
    // Add a user-facing message indicating a retry is being attempted for clarity
    const retryInitiationMessage: ChatMessage = {
        id: Date.now().toString() + '-retry-info',
        sender: 'user',
        text: `Retrying the question: "${chatItem.original_question}"`, 
        type: 'info',
        isLoadingResponse: false, // This isn't a message waiting for its own response, but an action log
    };
    setChatHistory(prev => [...prev, retryInitiationMessage]);

    const retryQuestion = `Regarding the question \"${chatItem.original_question}\": The previous response could be improved or I'm looking for an alternative. Please try answering it again, possibly with a different approach or more detail.`;
    await executeUserQuery(retryQuestion);
  };

  const handleFeedback = async (chatItem: ChatMessage, rating: 'positive' | 'negative') => {
    if (!chatItem.original_question) {
        console.warn("Cannot submit feedback: original question missing.");
        return;
    }

    // Mark feedback as given for the current item immediately for UI update (hides Like/Dislike)
    setChatHistory(prevChatHistory => 
      prevChatHistory.map(msg => 
        msg.id === chatItem.id ? { ...msg, feedback_given: true } : msg
      )
    );

    try {
      axios.post(`${API_URL}/feedback`, {
        question: chatItem.original_question,
        rating: rating,
        result: { 
          sql_query: chatItem.sql_query || undefined,
          analysis: chatItem.analysis || undefined,
          combined_analysis: chatItem.combined_analysis || undefined,
        }
      }).then(() => {
        console.log(`Feedback (${rating}) submitted for: ${chatItem.original_question}`);
        if (rating === 'positive') {
          fetchCacheSummary(); 
        }
      }).catch(err => {
        console.error("Error submitting feedback log:", err); 
      });

      // No alerts for the user here

    } catch (err) {
      console.error("Error in feedback handling logic:", err);
    }
  };

  // Render general content like DB info or schema error messages
  const renderGeneralContent = (content: any) => {
    if (content && typeof content === 'object' && content.error) {
        return <Alert variant="danger">{String(content.error)}</Alert>;
    }
    if (typeof content === 'object' && content !== null) {
      return <pre>{JSON.stringify(content, null, 2)}</pre>;
    }
    return <p>{String(content)}</p>;
  };

  // Specific renderer for Schema Data
  const renderSchemaData = (schemaData: SchemaData) => {
    return (
        <>
            {Object.entries(schemaData).map(([tableName, tableDetails]) => (
                <div key={tableName} className="mb-3">
                    <h6>{tableName}</h6>
                    <ListGroup variant="flush">
                        <ListGroup.Item>
                            <strong>Columns:</strong>
                            <ul>
                                {tableDetails.columns.map(col => (
                                    <li key={col.name}>{col.name} ({col.type})</li>
                                ))}
                            </ul>
                        </ListGroup.Item>
                        {/* Add foreign key rendering here if needed */}
                    </ListGroup>
                </div>
            ))}
        </>
    );
  };


  const renderCacheListItem = (item: ChatMessage) => (
    <ListGroup.Item key={item.id || item.cached_at} className="mb-2">
        <small className="text-muted">
            Cached at: {item.cached_at ? new Date(item.cached_at).toLocaleString() : 'N/A'} (Used: {item.usage_count || 0})
        </small>
        <p><strong>Q:</strong> {item.original_question || item.text}</p>
        {item.sql_query && <p><Badge bg="secondary">SQL</Badge> <code>{item.sql_query}</code></p>}
        {item.combined_analysis && <p><strong>Analysis:</strong> {item.combined_analysis}</p>}
        {item.visualization && (
            <div>
                <strong>Visualization:</strong>
                {typeof item.visualization === 'string' ? (
                    <p>{item.visualization}</p>
                ) : (
                    <p>[Chart data present - full rendering not shown in cache summary]</p>
                )}
            </div>
        )}
    </ListGroup.Item>
  );

  return (
    <Container fluid className="App vh-100 d-flex flex-column p-0">
      {/* Title Section - Using standard Row/Col with Bootstrap classes */}
      <Row className="bg-primary text-white p-4 text-center shadow-sm flex-shrink-0">
        <Col>
          <h1>AI Data Analyst</h1>
          <p className="lead mb-0">Turning Data Into Insights</p>
        </Col>
      </Row>

      <Row className="flex-grow-1" style={{ overflowY: 'hidden' }}>
        {/* Left Sidebar for Info */}
        <Col md={3} className="bg-light border-end p-3 d-flex flex-column" style={{ overflowY: 'auto', maxHeight: 'calc(100vh - 100px)' /* Adjusted: approx header height, can be fine-tuned */ }}>
          <h5 className="mb-3">Info Panel</h5>
          <Tab.Container id="sidebar-tab-info" activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'db')}>
            <Nav variant="tabs" fill className="mb-3 flex-shrink-0"> {/* Nav should not grow */}
              <Nav.Item><Nav.Link eventKey="db">DB Info</Nav.Link></Nav.Item>
              <Nav.Item><Nav.Link eventKey="schema">Schema</Nav.Link></Nav.Item>
              <Nav.Item><Nav.Link eventKey="cache">Cache</Nav.Link></Nav.Item>
            </Nav>
            {/* Tab.Content should grow and scroll if needed */}
            <Tab.Content className="flex-grow-1 overflow-auto" style={{ /* maxHeight can be used if needed, but flex-grow should handle it within fixed height Col */ }}> 
              <Tab.Pane eventKey="db" className="h-100">
                <h6>Database Connection</h6>
                {dbInfo ? renderGeneralContent(dbInfo) : <Spinner animation="border" size="sm" />}
              </Tab.Pane>
              <Tab.Pane eventKey="schema" className="h-100">
                <h6>Database Schema</h6>
                {schemaInfo ? (
                    schemaInfo.error ? <Alert variant="danger">{schemaInfo.error}</Alert> :
                    schemaInfo.data ? renderSchemaData(schemaInfo.data) :
                    <p>Schema not loaded or no tables found.</p>
                ) : <Spinner animation="border" size="sm" />}
              </Tab.Pane>
              <Tab.Pane eventKey="cache" className="h-100">
                <h6>Cache Summary</h6>
                {cacheSummary ? (
                  cacheSummary.error ? <Alert variant="danger">{cacheSummary.error}</Alert> :
                  <>
                    <p>Size: {cacheSummary.cache_size}, Hits: {cacheSummary.cache_hits}</p>
                    <ListGroup variant="flush">
                        {cacheSummary.cached_items && cacheSummary.cached_items.length > 0 ? 
                            cacheSummary.cached_items.map(renderCacheListItem) : <p>Cache is empty.</p>}
                    </ListGroup>
                  </>
                ) : <Spinner animation="border" size="sm" />}
              </Tab.Pane>
            </Tab.Content>
          </Tab.Container>
        </Col>

        {/* Main Chat Area */}
        <Col md={9} className="d-flex flex-column p-3">
          <div className="chat-history flex-grow-1 overflow-auto mb-3 p-2 border rounded" style={{maxHeight: 'calc(100vh - 200px)'}}> {/* Adjust max height as needed */}
            {chatHistory.map((chatItem) => (
              <div key={chatItem.id} className={`mb-3 p-2 rounded ${chatItem.sender === 'user' ? 'bg-primary text-white align-self-end' : 'bg-light align-self-start'}`} style={{ maxWidth: '85%', [chatItem.sender === 'user' ? 'marginLeft' : 'marginRight']: 'auto' }}>
                <strong>{chatItem.sender === 'user' ? 'You' : 'AI Analyst'}:</strong>
                {/* User's direct text message */}
                {chatItem.text && chatItem.sender === 'user' && (
                  <>
                    <p className="mb-1">{chatItem.text}</p>
                    {/* Per-message loading indicator */}
                    {chatItem.isLoadingResponse && (
                      <div className="mt-1 text-white-50">
                        <Spinner animation="border" size="sm" role="status" variant="light" className="me-1"/>
                        Thinking...
                      </div>
                    )}
                  </>
                )}
                
                {/* AI specific content rendering */}
                {chatItem.sender === 'ai' && (
                  <>
                    {/* AI's main textual response (analysis or combined_analysis) */}
                    {chatItem.text && <div className="mb-1"><ReactMarkdown>{chatItem.text}</ReactMarkdown></div>}
                    
                    {chatItem.sql_query && (
                      <Accordion className="mt-2 mb-1" /* defaultActiveKey="0" if you want it open by default */ >
                        <Accordion.Item eventKey="sql-query-item" style={{ border: 'none' }}>
                          <Accordion.Header bsPrefix="card-header py-1 px-2 rounded-top" style={{ backgroundColor: 'var(--bs-light)', borderBottom: '1px solid var(--bs-border-color)' }}>
                            <Badge bg="info">Generated SQL</Badge>
                          </Accordion.Header>
                          <Accordion.Body className="p-2 border border-top-0 rounded-bottom">
                            <pre className="mb-0" style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-all' }}><code>{chatItem.sql_query}</code></pre>
                          </Accordion.Body>
                        </Accordion.Item>
                      </Accordion>
                    )}
                    {chatItem.query_results && chatItem.query_results.length > 0 && (
                      <Card className="mt-2 mb-1">
                        <Card.Header bsPrefix="card-header py-1 px-2"><Badge bg="success">Query Results</Badge> ({chatItem.query_results.length} rows)</Card.Header>
                        <Card.Body className="p-0" style={{ maxHeight: '300px', overflowY: 'auto' }}>
                          <Table striped bordered hover size="sm" className="mb-0">
                            <thead style={{ backgroundColor: 'var(--bs-light)', position: 'sticky', top: 0, zIndex: 1 }}>
                              <tr>
                                {Object.keys(chatItem.query_results[0]).map(key => (
                                  <th key={key} style={{ position: 'sticky', top: 0, zIndex: 1, backgroundColor: 'var(--bs-light)' }}>
                                    {key}
                                  </th>
                                ))}
                              </tr>
                            </thead>
                            <tbody>
                              {chatItem.query_results.map((row, index) => (
                                <tr key={index}>
                                  {Object.values(row).map((val, i) => <td key={i}>{String(val)}</td>)}
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    )}
                    {/* MODIFIED VISUALIZATION RENDERING - Only show if it's an object (Plotly data) */}
                    {chatItem.visualization && typeof chatItem.visualization === 'object' && chatItem.visualization.data && chatItem.visualization.layout && (
                        <Card className="mt-2 mb-1">
                            <Card.Header bsPrefix="card-header py-1 px-2"><Badge bg="warning">Visualization</Badge></Card.Header>
                            <Card.Body className="p-2 text-center">
                                <Plot
                                    data={chatItem.visualization.data}
                                    layout={{ ...chatItem.visualization.layout, autosize: true }}
                                    useResizeHandler={true}
                                    style={{ width: '100%', minHeight: '300px' }}
                                    config={{ responsive: true }}
                                />
                            </Card.Body>
                        </Card>
                    )}
                    {/* END MODIFIED VISUALIZATION RENDERING */}
                  </>
                )}
                {chatItem.error && <Alert variant="danger" className="mt-2">Error: {chatItem.error}</Alert>}
                
                {/* Action Buttons for AI messages */}
                {chatItem.sender === 'ai' && chatItem.type === 'live' && !chatItem.error && (
                  <div className="mt-2 text-end">
                    <ButtonGroup size="sm">
                      {!chatItem.feedback_given && (
                        <>
                          <Button variant="outline-success" onClick={() => handleFeedback(chatItem, 'positive')} title="Like this response">
                            👍
                          </Button>
                          <Button variant="outline-danger" onClick={() => handleFeedback(chatItem, 'negative')} title="Dislike this response">
                            👎
                          </Button>
                        </>
                      )}
                      <Button variant="outline-info" onClick={() => handleRetry(chatItem)} title="Retry this question">
                        🔁 Retry
                      </Button>
                    </ButtonGroup>
                  </div>
                )}
              </div>
            ))}
            <div ref={chatEndRef} />
    </div>

          <Form onSubmit={handleSubmit} className="mt-auto p-2 border-top">
            <Row>
              <Col>
                <Form.Control
                  type="text"
                  value={userInput}
                  onChange={(e) => setUserInput(e.target.value)}
                  placeholder="Ask about your data..."
                  disabled={isLoading}
                  aria-label="User input"
                />
              </Col>
              <Col xs="auto">
                {isLoading ? (
                  <Button variant="danger" onClick={handleCancelQuery}>
                    Cancel
                  </Button>
                ) : (
                  <Button variant="primary" type="submit" disabled={isLoading}>
                    Send
                  </Button>
                )}
              </Col>
            </Row>
          </Form>
          {/* Global error display, if not tied to a specific chat message */}
          {error && 
            <Alert variant="danger" className="mt-2">{error}</Alert>}
        </Col>
      </Row>
    </Container>
  );
};

export default App;
