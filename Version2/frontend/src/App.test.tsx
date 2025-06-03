import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import App from './App';
import axios from 'axios';

// Mock axios
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock Recharts to prevent errors during tests as we are not testing the chart rendering itself here
jest.mock('recharts', () => {
  const OriginalRecharts = jest.requireActual('recharts');
  return {
    ...OriginalRecharts,
    ResponsiveContainer: ({ children }: { children: React.ReactNode }) => (
      <OriginalRecharts.ResponsiveContainer width={800} height={800}>
        {children}
      </OriginalRecharts.ResponsiveContainer>
    ),
  };
});


describe('App Component', () => {
  beforeEach(() => {
    // Reset mocks before each test
    mockedAxios.get.mockReset();
    mockedAxios.post.mockReset();

    // Default mock implementations
    mockedAxios.get.mockImplementation((url) => {
      if (url.includes('/db-info')) {
        return Promise.resolve({ data: { host: 'test-host', database: 'test-db', schema: 'test-schema' } });
      }
      if (url.includes('/schema-info')) {
        return Promise.resolve({ data: { 'table1': { columns: [{name: 'col1', type: 'text'}], foreign_keys: [] } } });
      }
      if (url.includes('/cache-summary')) {
        return Promise.resolve({ data: { cache_size: 0, cache_hits: 0, cached_items: [] } });
      }
      return Promise.reject(new Error('not found'));
    });

    mockedAxios.post.mockImplementation((url) => {
      if (url.includes('/query')) {
        return Promise.resolve({ 
          data: { 
            response: { 
              sql_query: 'SELECT 1', 
              analysis: 'Test analysis', 
              query_results: [{result: 'ok'}]
            }, 
            type: 'live' 
          } 
        });
      }
      if (url.includes('/feedback')) {
        return Promise.resolve({ data: { message: 'Feedback received' } });
      }
      return Promise.reject(new Error('not found'));
    });
  });

  test('renders navbar and main layout', async () => {
    render(<App />);
    expect(screen.getByText(/AI Data Analyst \(React\/Flask\)/i)).toBeInTheDocument();
    expect(screen.getByPlaceholderText(/Ask a question about your data.../i)).toBeInTheDocument();
    // Check for sidebar cards after data fetching
    await waitFor(() => expect(screen.getByText(/Database Connection/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/Schema Information/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/Response Cache Summary/i)).toBeInTheDocument());
  });

  test('fetches and displays db info, schema, and cache summary on load', async () => {
    render(<App />);
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:5001/db-info'));
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:5001/schema-info'));
    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledWith('http://localhost:5001/cache-summary'));

    // Check if some data from mocked calls is displayed
    await waitFor(() => expect(screen.getByText(/Host:/i)).toBeInTheDocument());
    // Check for table name from schema info
    await waitFor(() => expect(screen.getByText(/table1/i)).toBeInTheDocument()); 
    // More robust check for "Cached Responses: 0"
    await waitFor(() => {
      const strongElement = screen.getByText((content, node) => {
        return node?.tagName.toLowerCase() === 'strong' && content.startsWith('Cached Responses:');
      });
      expect(strongElement.parentElement?.textContent).toMatch(/Cached Responses:\s*0/i);
    });
  });

  test('submits a question and displays response', async () => {
    render(<App />);
    const input = screen.getByPlaceholderText(/Ask a question about your data.../i);
    const submitButton = screen.getByRole('button', { name: /Ask/i });

    fireEvent.change(input, { target: { value: 'What are the total sales?' } });
    fireEvent.click(submitButton);

    await waitFor(() => expect(mockedAxios.post).toHaveBeenCalledWith(
      'http://localhost:5001/query',
      { question: 'What are the total sales?' }
    ));
    
    // Check for parts of the mocked response
    await waitFor(() => expect(screen.getByText(/SQL Query:/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/SELECT 1/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/Test analysis/i)).toBeInTheDocument());
    await waitFor(() => expect(screen.getByText(/Query Results:/i)).toBeInTheDocument());
    // Check for a value from the query_results table
    await waitFor(() => expect(screen.getByText(/ok/i)).toBeInTheDocument()); 
  });

  test('submits feedback for a response', async () => {
  render(<App />);
    // First, submit a question to get a response in history
    const input = screen.getByPlaceholderText(/Ask a question about your data.../i);
    const submitButton = screen.getByRole('button', { name: /Ask/i });
    fireEvent.change(input, { target: { value: 'Test feedback q' } });
    fireEvent.click(submitButton);

    // Wait for the response to appear and then the feedback buttons
    await waitFor(() => expect(screen.getByText(/SQL Query:/i)).toBeInTheDocument());
    const helpfulButton = await screen.findByRole('button', { name: /^👍 Helpful$/i });
    
    // Mock window.alert
    const alertMock = jest.spyOn(window, 'alert').mockImplementation(() => {});

    fireEvent.click(helpfulButton);

    await waitFor(() => expect(mockedAxios.post).toHaveBeenCalledWith(
      'http://localhost:5001/feedback',
      expect.objectContaining({ rating: 'positive', question: 'Test feedback q' })
    ));
    expect(alertMock).toHaveBeenCalledWith('Feedback (positive) submitted!');
    alertMock.mockRestore();
  });

});
