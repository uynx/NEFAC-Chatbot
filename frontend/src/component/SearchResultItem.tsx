import React from "react";

interface SearchResult {
  title: string;
  link: string;
  type: string;
  timestamp_seconds?: number;
  summary?: string;
  content?: string; // The actual chunk content from transcript
}

export const SearchResultItem: React.FC<{ result: SearchResult}> = ({ result }) => {
  // Helper function to get the appropriate tag info based on type
  const getTagInfo = (type: string, link: string) => {
    // First check the backend-provided type field
    if (type === 'youtube') {
      return {
        label: 'YouTube',
        className: 'px-2 py-1 text-xs font-medium rounded-full bg-red-400 text-white'
      };
    }
    
    // Fall back to URL-based detection
    if (link && link.toLowerCase().includes('youtube')) {
      return {
        label: 'YouTube',
        className: 'px-2 py-1 text-xs font-medium rounded-full bg-red-400 text-white'
      };
    } else if (link && link.toLowerCase().includes('nefac.org')) {
      return {
        label: 'NEFAC Website',
        className: 'px-2 py-1 text-xs font-medium rounded-full bg-blue-500 text-white'
      };
    } else {
      // Default for any other type (pdf, etc.)
      return {
        label: 'NEFAC Website',
        className: 'px-2 py-1 text-xs font-medium rounded-full bg-blue-500 text-white'
      };
    }
  };

  const tagInfo = getTagInfo(result.type, result.link);

  return (
    <div className="p-4 border-l-4 border-blue-200 bg-gray-50 rounded-r-lg">
      {/* Source info header */}
      <div className="flex items-start justify-between mb-3">
        <div className="flex-1">
          <h3 className="text-base font-semibold text-gray-900 mb-1">{result.title}</h3>
          <a 
            href={result.link} 
            target="_blank" 
            rel="noopener noreferrer"
            className="text-sm text-blue-600 hover:text-blue-800 hover:underline"
          >
            {result.link}
            {result.timestamp_seconds && (
              <span className="text-gray-500 ml-1">
                (at {Math.floor(result.timestamp_seconds / 60)}:{(result.timestamp_seconds % 60).toString().padStart(2, '0')})
              </span>
            )}
          </a>
        </div>
        
        {/* Tag */}
        <span className={tagInfo.className}>
          {tagInfo.label}
        </span>
      </div>
      
      {/* Transcript content */}
      {result.content && (
        <div className="mt-3">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Relevant Content:</h4>
          <div className="bg-white p-3 rounded border-l-2 border-blue-300">
            <p className="text-sm text-gray-700 leading-relaxed italic">
              "{result.content}"
            </p>
          </div>
        </div>
      )}
      
      {/* Summary if available and different from content */}
      {result.summary && result.summary !== result.content && (
        <div className="mt-3">
          <h4 className="text-sm font-medium text-gray-700 mb-2">Summary:</h4>
          <p className="text-sm text-gray-600">
            {result.summary}
          </p>
        </div>
      )}
    </div>
  );
};