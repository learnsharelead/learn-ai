# Apple-Inspired Website Design System & Content Architecture

## Core Design Philosophy

**"Simplicity is the ultimate sophistication"** — Design with ruthless focus on what matters. Every element must earn its place. Remove until it breaks, then add back one element.

---

## Visual Design Principles

### Typography System

**Primary Typeface:** San Francisco Display / SF Pro (or alternatives: Inter, Helvetica Neue)

**Type Scale:**
- Hero Headline: 80-120px, weight 700, tracking -2%
- Page Headline: 48-64px, weight 600, tracking -1%
- Section Headline: 32-40px, weight 600
- Subsection: 24-28px, weight 600
- Body Large: 21px, weight 400, line-height 1.5
- Body: 17px, weight 400, line-height 1.6
- Caption: 14px, weight 400, line-height 1.4

**Typography Rules:**
- Maximum 2-3 font weights per page
- Generous line height (1.5-1.6 for readability)
- Letter spacing: tighter for headlines, comfortable for body
- Never use pure black (#000) — use #1d1d1f instead

### Color Philosophy

**Primary Palette:**
- Background: Pure white (#ffffff) or near-black (#000000)
- Text Primary: #1d1d1f (deep gray-black)
- Text Secondary: #6e6e73 (medium gray)
- Accent: One brand color used sparingly
- Links: #0066cc (Apple blue) or brand color

**Secondary Palette:**
- Success: #34c759
- Warning: #ff9500
- Error: #ff3b30
- Subtle borders: #d2d2d7

**Gradient Usage:**
- Subtle, sophisticated gradients for product showcases
- Never garish or oversaturated
- Often used in hero sections with product imagery

### Spacing System (8-Point Grid)

**Micro Spacing:**
- 4px: Icon-to-text
- 8px: Related elements
- 16px: Component internal padding
- 24px: Between related sections

**Macro Spacing:**
- 32px: Between components
- 48px: Between major sections (mobile)
- 80px: Between major sections (desktop)
- 120px: Between hero and next section

### Layout Principles

**Grid System:**
- Container max-width: 1440px
- Content max-width: 980px for text
- Column gutters: 24px (mobile), 32px (desktop)
- Edge margins: 20px (mobile), 40px (tablet), 80px (desktop)

**White Space:**
- Embrace emptiness — it creates luxury
- Content should breathe
- Never crowd elements
- Asymmetry is powerful when intentional

---

## Navigation Architecture

### Primary Navigation (Global Header)

**Structure:**
```
Logo | Product | Solutions | Resources | About | Support | Search | Cart | Account
```

**Specifications:**
- Height: 44px (mobile), 48px (desktop)
- Background: Translucent blur (rgba(255,255,255,0.8)) with backdrop-filter
- Position: Sticky with slide-up animation on scroll down
- Typography: 14px, weight 400, letter-spacing 0.5px
- Spacing: 32px between nav items

**Behavior:**
- Mega menu appears on hover (not click)
- 200ms fade-in animation
- Mega menu has subtle drop shadow
- Current page indicator: subtle underline or bold weight

### Mega Menu Structure

#### PRODUCT Menu
**Layout:** 3-column grid with featured product imagery

**Column 1: Product Categories**
- All Products (overview link)
- [Product Line 1]
  - Model A
  - Model B
  - Model C
- [Product Line 2]
  - Model X
  - Model Y
- Accessories
- Compare Products (link)

**Column 2: Features**
- What's New (featured badge)
- Product Highlights
- Technical Specs
- Why Choose Us
- Product Videos

**Column 3: Visual**
- Hero product image
- "Shop Now" CTA
- Starting at $XXX (pricing)
- Available in X colors

#### SOLUTIONS Menu
**Layout:** 2-column with icon-based navigation

**Column 1: By Industry**
- 🏢 Enterprise
- 🎓 Education
- 🏥 Healthcare
- 💼 Small Business
- 🏠 Home Use

**Column 2: By Use Case**
- Creative Professional
- Data & Analytics
- Development
- Collaboration
- Security & Privacy

**Bottom Row:**
- Customer Stories
- Case Studies
- ROI Calculator

#### RESOURCES Menu
**Layout:** 4-column grid, compact

**Learn**
- Getting Started
- Tutorials
- Video Guides
- Webinars
- Documentation
- Best Practices

**Community**
- Forums
- User Groups
- Events
- Blog
- Newsletter

**Downloads**
- Software Updates
- Drivers
- Manuals
- Wallpapers
- Assets

**Support**
- Contact Us
- FAQs
- Service Status
- Feedback

#### ABOUT Menu
**Layout:** Simple single column

- Our Story
- Leadership Team
- Careers (We're Hiring badge)
- Press & Media
- Investors
- Sustainability
- Privacy Policy
- Newsroom

#### SUPPORT Menu
**Layout:** Split layout with search

**Quick Access:**
- Contact Support (primary CTA)
- Live Chat
- Schedule Callback
- Service & Repair
- Warranty Info

**Self-Service:**
- Search support articles
- Community Forums
- Video Tutorials
- Download Manuals
- Check Order Status

---

## Content Hierarchy & Page Structure

### Homepage Structure

**1. Hero Section (Full viewport height)**
- Large product image or video (autoplay, muted, loop)
- Headline (one powerful statement)
- Subheadline (brief, compelling)
- Two CTAs: "Learn More" (ghost) + "Buy Now" (filled)
- Scroll indicator (subtle down arrow)

**2. Product Showcase 1 (Text Left, Image Right)**
- Section headline
- Body copy (2-3 sentences max)
- Feature bullets (3-4 items with icons)
- "Learn more >" link
- Product image with soft shadow

**3. Product Showcase 2 (Image Left, Text Right)**
- Mirror layout of previous section
- Different product/feature
- Maintain visual rhythm

**4. Video Section (Full-width)**
- Background video or interactive demo
- Minimal text overlay
- "Watch the film" CTA

**5. Feature Grid (3 columns)**
- Icon + Headline + Short description
- Examples: Performance | Privacy | Ecosystem
- Links to detailed pages

**6. Testimonials / Social Proof**
- Large quotes in center
- Customer photos (circular)
- Company logos of clients
- Star ratings if applicable

**7. CTA Section (Full-width, colored background)**
- Bold headline
- Single clear action
- Generous padding
- High contrast

**8. Footer (Comprehensive)**
- See detailed footer structure below

---

### Product Page Structure

**Section 1: Hero**
- Product name (large)
- Tagline
- Price (if applicable)
- Color selector (visual swatches)
- "Add to Cart" + "Learn More"
- Large product image (rotatable 360°)

**Section 2: Key Features (Scrollytelling)**
- Sticky image on left
- Features reveal on right as you scroll
- Smooth animations
- Each feature = headline + paragraph + visual

**Section 3: Technical Specs (Tabs)**
- Overview | Specifications | In the Box | Support
- Clean table layout
- Collapsible sections
- Download spec sheet link

**Section 4: Compare Models**
- Side-by-side comparison table
- Highlight differences
- "Choose yours" CTAs

**Section 5: Accessories**
- "You might also like"
- Product cards (4 across)
- Quick add to cart

**Section 6: Reviews**
- Star rating summary
- Filter by rating
- Verified purchase badges
- Helpful voting system

---

### Solutions/Category Page

**Hero:**
- Category name
- Brief description
- Filter/Sort bar (sticky)

**Content Grid:**
- Card-based layout (3-4 columns)
- Image + headline + price
- Hover: Quick view button appears
- "Load more" or infinite scroll

**Sidebar Filters (Desktop):**
- Price range slider
- Categories (checkboxes)
- Brand/Type
- Color swatches
- Ratings
- "Clear all" link

---

### About/Story Page

**Layout:**
- Long-form reading experience
- Max width: 780px
- Large imagery between sections
- Pull quotes in larger type
- Timeline visualization
- Team member grid with photos

---

### Support/Help Center

**Hero with Search:**
- "How can we help you?"
- Large search bar
- Popular topics below

**Content:**
- Categorized articles
- Video tutorials
- Community forums
- Contact options prominently placed

---

## Component Specifications

### Buttons

**Primary Button:**
- Background: Brand color or #0071e3
- Text: White, 14px, weight 400
- Padding: 12px 24px
- Border-radius: 980px (pill shape)
- Hover: Slight darken (10%)
- Transition: 200ms ease

**Secondary Button (Ghost):**
- Background: Transparent
- Border: 1px solid current color
- Text: Brand color or black
- Same sizing as primary
- Hover: Fill with light background

**Text Link:**
- Color: #0066cc or brand color
- No underline by default
- Underline on hover
- Arrow after text: "Learn more >"

### Cards

**Product Card:**
- White background
- Subtle shadow on hover
- Image aspect ratio: 1:1 or 4:3
- Padding: 20px
- Border-radius: 18px
- Title: 17px, weight 600
- Price: 14px, weight 400, gray
- Hover: translateY(-4px) + deeper shadow

**Content Card:**
- Image on top (full-width)
- Title + description
- No borders, clean
- "Read more" link at bottom

### Forms

**Input Fields:**
- Height: 44px
- Border: 1px solid #d2d2d7
- Border-radius: 8px
- Padding: 0 16px
- Font-size: 17px
- Focus: Blue border, subtle shadow
- Label: Float above when filled

**Validation:**
- Inline error messages (red text below)
- Success checkmark (green)
- Real-time validation on blur

### Icons

**Style:**
- Outlined (2px stroke)
- 24x24px standard size
- Consistent corner radius
- SF Symbols style or similar
- Monochrome with theme color

### Images

**Treatment:**
- High resolution (2x)
- Lazy loading
- WebP format with fallbacks
- Subtle shadows under product images
- Transparent backgrounds when possible

---

## Animation & Interaction

### Micro-interactions

**Hover States:**
- Scale: 1.02-1.05 (subtle)
- Duration: 200-300ms
- Easing: ease-out
- Cursor: pointer on interactive elements

**Page Transitions:**
- Fade in: 400ms
- Slide up on scroll reveal: 600ms, stagger by 100ms
- No jarring movements

**Loading:**
- Skeleton screens (not spinners)
- Progress bars for known durations
- Optimistic UI updates

**Scroll Behavior:**
- Smooth scroll (CSS: scroll-behavior: smooth)
- Parallax effects (subtle, 0.5x speed max)
- Sticky elements that reveal/hide intelligently
- Intersection Observer for reveal animations

---

## Responsive Breakpoints

**Mobile First Approach:**

**Small Mobile:** 320px - 374px
- Single column
- Stacked navigation (hamburger)
- Touch-friendly targets (44px min)
- Reduced text sizes

**Mobile:** 375px - 767px
- 20px side margins
- Full-width images
- Collapsible sections
- Bottom navigation for key actions

**Tablet:** 768px - 1023px
- 2-column grids
- Persistent navigation
- 40px side margins
- Larger typography

**Desktop:** 1024px - 1439px
- Full multi-column layouts
- Hover states active
- Mega menus
- 60px side margins

**Large Desktop:** 1440px+
- Max container width enforced
- Generous spacing
- Full resolution images
- 80px+ side margins

---

## Accessibility Standards

**Must-Haves:**
- WCAG 2.1 Level AA minimum
- Keyboard navigation (visible focus)
- Screen reader optimized (ARIA labels)
- Color contrast: 4.5:1 for text
- Skip to main content link
- Alt text for all images
- Captions for videos
- Reduced motion media query

**Focus States:**
- Blue outline: 2px solid #0066cc
- Offset: 2px
- Border-radius matches element

---

## Footer Architecture

**Layout:** 5-column grid (mobile: accordion)

**Column 1: Product**
- Product Line 1
- Product Line 2
- Accessories
- Gift Cards
- Marketplace

**Column 2: Support**
- Contact Support
- Order Status
- Returns
- Warranty
- FAQs
- Manuals

**Column 3: Company**
- About Us
- Careers
- Press
- Investors
- Events
- Blog

**Column 4: Legal**
- Privacy Policy
- Terms of Service
- Cookie Policy
- Accessibility
- Sitemap

**Column 5: Connect**
- Newsletter signup
- Social icons (monochrome)
- App download links

**Bottom Row:**
- Copyright notice
- Country/Region selector
- Payment method icons

---

## Content Writing Guidelines

**Voice & Tone:**
- Clear, not clever
- Confident, not boastful
- Simple, not simplistic
- Friendly, not overfamiliar

**Sentence Structure:**
- Short sentences (15-20 words max)
- Active voice
- One idea per sentence
- Front-load important information

**Formatting:**
- Headlines ask questions or make bold statements
- Use parallel structure in lists
- Numbers as numerals (not spelled out)
- Sentence case for headlines, not title case

**Call-to-Actions:**
- Action-oriented verbs
- "Learn more" not "Click here"
- "Shop iPhone" not "Click to shop"
- Create urgency without being pushy

---

## Performance Requirements

**Speed Targets:**
- First Contentful Paint: <1.8s
- Time to Interactive: <3.9s
- Lighthouse Score: 90+

**Optimization:**
- Lazy load images below fold
- Code splitting
- CDN for assets
- Minified CSS/JS
- HTTP/2 or HTTP/3
- Cached static assets

---

## Examples of Apple-Like Content Flow

### Product Landing Page Example

```
HERO SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
iPhone Pro. So powerful. So priced right.
From $999 or $41.62/mo.
[Learn more]  [Buy]
[Large product image]

FEATURE 1 (Dark background)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A camera that captures magic.
[Product image]
Advanced triple-camera system.
Night mode on every camera.
Up to 3x optical zoom.

FEATURE 2 (White background)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Performance that breaks records.
[Chip visualization]
A17 Pro chip.
The fastest we've ever made.
20% faster than the competition.

VIDEO SECTION
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Full-width background video]
See it in action.
[Play button]

COMPARISON
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Which iPhone is right for you?
[Side-by-side comparison table]

ACCESSORIES
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Complete your iPhone.
[4-column grid of cases, chargers, etc.]
```

---

## Final Checklist

**Before Launch:**
- [ ] Test on iPhone, iPad, Mac Safari
- [ ] Verify all animations are 60fps
- [ ] Check every link works
- [ ] Validate HTML/CSS
- [ ] Test keyboard navigation
- [ ] Run Lighthouse audit
- [ ] Verify all images have alt text
- [ ] Test form submissions
- [ ] Check analytics tracking
- [ ] Review legal pages
- [ ] Test payment flow
- [ ] Verify responsive breakpoints

---

## Inspiration Resources

**Study These Apple Pages:**
- apple.com/iphone (product showcase mastery)
- apple.com/environment (storytelling)
- apple.com/retail (service design)
- developer.apple.com (technical content)

**Key Takeaways:**
- Restraint is power
- Every word matters
- White space is content
- Performance is design
- Accessibility is non-negotiable
