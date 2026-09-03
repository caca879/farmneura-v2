import React, { useState } from 'react';
import { Sprout, Lock, Mail, User as UserIcon, Eye, EyeOff, LogIn, UserPlus, AlertCircle, CheckCircle2 } from 'lucide-react';
import { translations } from '../utils/translations';
import { loginUser, signupUser } from '../services/api';

export default function Auth({ onAuthSuccess, lang, initialTab = 'login' }) {
  const [isLogin, setIsLogin] = useState(initialTab === 'login');
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Form State
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');

  const t = translations[lang] || translations["Bahasa Melayu"];

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setSuccess('');

    // Validations
    if (!email.trim() || !password.trim()) {
      setError(lang.includes('Melayu') ? 'Sila isi emel dan kata laluan.' : 'Please enter email and password.');
      return;
    }

    if (!isLogin) {
      if (!fullName.trim()) {
        setError(lang.includes('Melayu') ? 'Sila isi nama penuh anda.' : 'Please enter your full name.');
        return;
      }
      if (password !== confirmPassword) {
        setError(lang.includes('Melayu') ? 'Kata laluan tidak sepadan.' : 'Passwords do not match.');
        return;
      }
      if (password.length < 6) {
        setError(lang.includes('Melayu') ? 'Kata laluan mestilah sekurang-kurangnya 6 aksara.' : 'Password must be at least 6 characters.');
        return;
      }
    }

    setLoading(true);

    try {
      if (isLogin) {
        const res = await loginUser({ email, password });
        setSuccess(t.authSuccessMsg);
        localStorage.setItem('farmneura_user', JSON.stringify(res.user));
        localStorage.setItem('farmneura_token', res.access_token);
        setTimeout(() => {
          onAuthSuccess(res.user);
        }, 800);
      } else {
        const res = await signupUser({
          full_name: fullName,
          email,
          password
        });
        setSuccess(t.signupSuccessMsg);
        localStorage.setItem('farmneura_user', JSON.stringify(res.user));
        localStorage.setItem('farmneura_token', res.access_token);
        setTimeout(() => {
          onAuthSuccess(res.user);
        }, 800);
      }
    } catch (err) {
      const msg = err.response?.data?.detail || (lang.includes('Melayu') ? 'Gagal memproses permintaan.' : 'Failed to process request.');
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-md mx-auto my-6 bg-white border border-gray-200 rounded-3xl shadow-xl overflow-hidden">
      {/* Header Banner */}
      <div className="bg-gradient-to-r from-emerald-900 via-emerald-800 to-amber-700 p-6 text-white text-center border-b border-emerald-700">
        <div className="inline-flex items-center justify-center p-3 bg-black/20 rounded-2xl backdrop-blur-md mb-3 border border-white/10">
          <Sprout className="w-8 h-8 text-emerald-300" />
        </div>
        <h2 className="text-xl font-black tracking-tight text-white drop-shadow-md">
          {isLogin ? t.loginTitle : t.signupTitle}
        </h2>
        <p className="text-xs text-emerald-100 mt-1 font-medium drop-shadow-xs">
          {t.authSubtitle}
        </p>
      </div>

      {/* Auth Tab Switcher */}
      <div className="flex border-b border-gray-200 bg-gray-50">
        <button
          onClick={() => { setIsLogin(true); setError(''); setSuccess(''); }}
          className={`flex-1 py-3 text-xs font-bold transition flex items-center justify-center space-x-1.5 ${
            isLogin ? 'bg-white text-emerald-950 border-b-2 border-emerald-600 shadow-xs font-black' : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          <LogIn className="w-4 h-4" />
          <span>{t.tabLogin}</span>
        </button>
        <button
          onClick={() => { setIsLogin(false); setError(''); setSuccess(''); }}
          className={`flex-1 py-3 text-xs font-bold transition flex items-center justify-center space-x-1.5 ${
            !isLogin ? 'bg-white text-emerald-950 border-b-2 border-emerald-600 shadow-xs font-black' : 'text-gray-500 hover:text-gray-700'
          }`}
        >
          <UserPlus className="w-4 h-4" />
          <span>{t.tabSignup}</span>
        </button>
      </div>

      {/* Form Content */}
      <div className="p-6 space-y-4">
        {error && (
          <div className="bg-red-50 border-l-4 border-red-500 p-3 rounded-xl text-xs text-red-700 font-semibold flex items-center space-x-2 animate-fadeIn">
            <AlertCircle className="w-4 h-4 text-red-500 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {success && (
          <div className="bg-emerald-50 border-l-4 border-emerald-500 p-3 rounded-xl text-xs text-emerald-700 font-semibold flex items-center space-x-2 animate-fadeIn">
            <CheckCircle2 className="w-4 h-4 text-emerald-500 flex-shrink-0" />
            <span>{success}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          {!isLogin && (
            <div>
              <label className="block text-xs font-bold text-gray-700 mb-1">{t.fullNameLabel}</label>
              <div className="relative">
                <UserIcon className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
                <input
                  type="text"
                  required
                  placeholder="e.g. Ahmad Petani"
                  value={fullName}
                  onChange={(e) => setFullName(e.target.value)}
                  className="w-full pl-9 pr-3 py-2.5 bg-gray-50 border border-gray-300 rounded-xl text-xs font-semibold text-gray-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-600 transition"
                />
              </div>
            </div>
          )}

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1">{t.emailLabel}</label>
            <div className="relative">
              <Mail className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
              <input
                type="email"
                required
                placeholder="petani@farmneura.com"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full pl-9 pr-3 py-2.5 bg-gray-50 border border-gray-300 rounded-xl text-xs font-semibold text-gray-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-600 transition"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-bold text-gray-700 mb-1">{t.passwordLabel}</label>
            <div className="relative">
              <Lock className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
              <input
                type={showPassword ? "text" : "password"}
                required
                placeholder="••••••••"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full pl-9 pr-10 py-2.5 bg-gray-50 border border-gray-300 rounded-xl text-xs font-semibold text-gray-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-600 transition"
              />
              <button
                type="button"
                onClick={() => setShowPassword(!showPassword)}
                className="absolute right-3 top-3 text-gray-400 hover:text-gray-600 focus:outline-none"
              >
                {showPassword ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
              </button>
            </div>
          </div>

          {!isLogin && (
            <div>
              <label className="block text-xs font-bold text-gray-700 mb-1">{t.confirmPasswordLabel}</label>
              <div className="relative">
                <Lock className="w-4 h-4 text-gray-400 absolute left-3 top-3" />
                <input
                  type={showPassword ? "text" : "password"}
                  required
                  placeholder="••••••••"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="w-full pl-9 pr-10 py-2.5 bg-gray-50 border border-gray-300 rounded-xl text-xs font-semibold text-gray-800 focus:bg-white focus:outline-none focus:ring-2 focus:ring-emerald-600 transition"
                />
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={loading}
            className={`w-full py-3.5 rounded-xl font-black text-xs text-white shadow-md hover:shadow-lg transition flex items-center justify-center space-x-2 ${
              loading ? 'bg-gray-400 cursor-not-allowed' : 'bg-emerald-700 hover:bg-emerald-800'
            }`}
          >
            {isLogin ? <LogIn className="w-4 h-4" /> : <UserPlus className="w-4 h-4" />}
            <span>{loading ? (lang.includes('Melayu') ? 'Memproses...' : 'Processing...') : (isLogin ? t.btnLogin : t.btnSignup)}</span>
          </button>
        </form>

        <div className="pt-2 text-center">
          <button
            onClick={() => { setIsLogin(!isLogin); setError(''); setSuccess(''); }}
            className="text-xs font-bold text-emerald-700 hover:text-emerald-900 hover:underline focus:outline-none"
          >
            {isLogin 
              ? (lang.includes('Melayu') ? 'Belum ada akaun? Daftar di sini →' : 'No account yet? Register here →')
              : (lang.includes('Melayu') ? 'Sudah ada akaun? Log masuk →' : 'Already have an account? Log in →')}
          </button>
        </div>
      </div>
    </div>
  );
}
