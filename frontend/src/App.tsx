import { AnalyzePage } from "./components/AnalyzePage";

function App() {
  return (
    <div className="min-h-screen bg-bg">
      <header className="border-b border-border px-6 py-4">
        <h1 className="text-xl font-bold text-text tracking-tight">
          Chord Analysis
        </h1>
      </header>
      <main className="max-w-6xl mx-auto p-6">
        <AnalyzePage />
      </main>
    </div>
  );
}

export default App;
