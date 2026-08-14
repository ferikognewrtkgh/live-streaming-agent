import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  metadataBase: new URL(
    process.env.NEXT_PUBLIC_SITE_URL ?? "http://localhost:3000",
  ),
  title: "Live Streaming Agent 控制台",
  description:
    "Live Streaming Agent 的独立模型配置、知识库、直播互动与历史对话测试控制台。",
  openGraph: {
    title: "Live Streaming Agent 控制台",
    description: "独立后端与 Elasticsearch 历史对话测试控制台",
    type: "website",
    images: [{ url: "/og.png", width: 1734, height: 907 }],
  },
  twitter: {
    card: "summary_large_image",
    title: "Live Streaming Agent 控制台",
    description: "独立后端与 Elasticsearch 历史对话测试控制台",
    images: ["/og.png"],
  },
  icons: {
    icon: "/favicon.svg",
    shortcut: "/favicon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="zh-CN">
      <body className={`${geistSans.variable} ${geistMono.variable}`}>
        {children}
      </body>
    </html>
  );
}
