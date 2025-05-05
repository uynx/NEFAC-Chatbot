import React from "react";
import { useEffect } from 'react';

interface SearchResult {
  title: string;
  link: string;
  audience: string[];
  nefac_category: string[];
  resource_type: string[];
  chunks: { summary: string; citations: any[] }[]; // Adjust as needed
}

export const SearchResultItem: React.FC<{ result: SearchResult}> = ({ result }) => {
  // Get the first two tags from each array
  const audienceTags = result.audience.slice(0, 2);
  const categoryTags = result.nefac_category.slice(0, 2);
  const resourceTags = result.resource_type.slice(0, 2);

  return (
    <div className="p-4 border rounded-lg shadow-sm hover:shadow-md transition-shadow">
      {/* Title */}
      <h3 className="text-lg font-semibold">{result.title}</h3>
      
      {/* Link */}
      <a href={result.link} className="text-blue-500 hover:underline">
      {result.link}
        </a>
      
      {/* Tags under the link */}
      <div className="mt-2 flex flex-wrap gap-2">
        {/* Audience tags in yellow */}
        {audienceTags.map((tag, index) => (
          <span
            key={`audience-${index}`}
            className="px-2 py-1 text-xs font-medium rounded-full bg-yellow-500 text-black"
          >
            {tag}
          </span>
        ))}
        
        {/* NEFAC Category tags in green */}
        {categoryTags.map((tag, index) => (
          <span
            key={`category-${index}`}
            className="px-2 py-1 text-xs font-medium rounded-full bg-green-500 text-white"
          >
            {tag}
          </span>
        ))}
        
        {/* Resource Type tags in purple */}
        {resourceTags.map((tag, index) => (
          <span
            key={`resource-${index}`}
            className="px-2 py-1 text-xs font-medium rounded-full bg-purple-500 text-white"
          >
            {tag}
          </span>
        ))}
      </div>
      
      {/* Chunks (if any) */}
      {result.chunks.map((chunk, index) => (
        <p key={index} className="mt-2 text-gray-700">
          {chunk.summary}
        </p>
      ))}
    </div>
  );
};